"""验证本地标注身份、错误输入及跨生成器文件名消歧。"""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import uuid4
import pandas as pd
import data_io
import extract_features


class LocalDataTests(unittest.TestCase):
    def setUp(self):
        self.folder = data_io.ROOT/('.gc-ridge-test-'+uuid4().hex)
        self.folder.mkdir()
        (self.folder/'configs').mkdir()
        self.frame = pd.DataFrame({'sample_id':['test-a','test-b','test-c'],'mos':[0.2,0.7,0.5]})
        record = {'example':{'sha256_at_12_decimals':data_io.target_fingerprint(self.frame)}}
        (self.folder/'configs/target_fingerprints.json').write_text(json.dumps(record))

    def tearDown(self):
        self.assertEqual(self.folder.resolve().parent,data_io.ROOT.resolve())
        for path in sorted(self.folder.rglob('*'),key=lambda p:len(p.parts),reverse=True):
            self.assertTrue(path.resolve().is_relative_to(self.folder.resolve()))
            if path.is_file(): path.unlink()
            else: path.rmdir()
        self.folder.rmdir()

    def validate(self, frame):
        with patch.object(data_io,'ROOT',self.folder), patch.object(data_io,'split_ids',return_value=self.frame[['sample_id']]):
            return data_io.validate_labels(frame,'example')

    def test_labels_join_by_id_not_row_order(self):
        result = self.validate(self.frame.iloc[::-1])
        self.assertEqual(result.to_dict('list'),self.frame.to_dict('list'))

    def test_wrong_quality_values_fail_fingerprint(self):
        frame = self.frame.copy()
        frame.loc[0,'mos'] = 0.3
        with self.assertRaisesRegex(ValueError,'fingerprint'): self.validate(frame)

    def test_duplicate_or_missing_labels_fail(self):
        for frame in [self.frame.iloc[[0,0,2]],self.frame.iloc[:2]]:
            with self.assertRaises(ValueError): self.validate(frame)

    def test_missing_local_data_fails_without_fallback(self):
        with self.assertRaises(FileNotFoundError): data_io.read_labels(self.folder,'AIGCIQA2023')

    def test_generator_layout_disambiguates_same_filename(self):
        for name in ['Gen-A','GenB']:
            (self.folder/name).mkdir()
            (self.folder/name/'001-1.png').write_bytes(b'path resolver fixture')
        rows = pd.DataFrame({'sample_id':['id-a','id-b'],'official_filename':['001-1.png']*2,'client_id':['gena','genb']})
        paths = extract_features.image_paths(rows,self.folder,'generator')
        self.assertEqual([p.parent.name for p in paths],['Gen-A','GenB'])
        with self.assertRaises(ValueError): extract_features.image_paths(rows,self.folder,'flat')

    def test_indexed_layout_uses_global_image_index(self):
        (self.folder/'7.png').write_bytes(b'path resolver fixture')
        rows = pd.DataFrame({'sample_id':['aigciqa2023_Test_7'],'official_filename':['001-1.png'],'client_id':['Test']})
        self.assertEqual(extract_features.image_paths(rows,self.folder,'indexed')[0].name,'7.png')


if __name__ == '__main__':
    unittest.main()
