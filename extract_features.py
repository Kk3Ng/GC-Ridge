"""用官方 OpenAI CLIP 从本地图像提取特征；不包含图像或编码器权重。"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import numpy as np
from data_io import ROOT, SLUGS, split_ids, feature_fingerprint
from reproduce import output_directory, write_json

MODELS = {
    'RN50': ('RN50','afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762'),
    'ViT-B-32': ('ViT-B/32','40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'),
}


def generator_key(value):
    return re.sub('[^a-z0-9]','',str(value).casefold())


def image_paths(rows, directory, layout):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError('The local image root does not exist.')
    files = {}
    for path in directory.rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.png','.jpg','.jpeg'}:
            files.setdefault(path.name,[]).append(path)
    result = []
    for row in rows.itertuples():
        name = f'{row.sample_id.rsplit("_",1)[-1]}.png' if layout == 'indexed' else row.official_filename
        candidates = files.get(name,[])
        if layout == 'generator':
            candidates = [p for p in candidates if any(generator_key(part)==generator_key(row.client_id)
                          for part in p.relative_to(directory).parts[:-1])]
        if len(candidates) != 1:
            raise ValueError(f'{row.sample_id}: expected one image for the selected layout; found {len(candidates)}.')
        if not candidates[0].resolve().is_relative_to(directory):
            raise ValueError('Image symlinks must remain inside the requested image root.')
        result.append(candidates[0])
    return result


def encode_images(paths, backbone, checkpoint=None, device='auto', batch_size=32, cache_dir=None):
    if batch_size < 1:
        raise ValueError('batch_size must be positive.')
    import torch
    import torchvision
    import clip
    from PIL import Image, __version__ as pillow_version
    model_name, expected_hash = MODELS[backbone]
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected_hash:
            raise ValueError('Checkpoint checksum does not match the study encoder.')
        model_name = str(checkpoint)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # clip.load 的命名模型从其官方地址下载并校验；显式本地权重先通过上面的哈希检查。
    model, preprocess = clip.load(model_name,device=device,jit=False,download_root=str(cache_dir) if cache_dir else None)
    model.eval()
    blocks = []
    with torch.no_grad():
        for start in range(0,len(paths),batch_size):
            tensors = []
            for path in paths[start:start+batch_size]:
                with Image.open(path) as img:
                    tensors.append(preprocess(img.convert('RGB')))
            x = model.encode_image(torch.stack(tensors).to(device)).float()
            x = x/x.norm(dim=-1,keepdim=True)
            blocks.append(x.cpu().numpy().astype(np.float32))
            print(f'Encoded {min(start+batch_size,len(paths))}/{len(paths)} images',flush=True)
    features = np.vstack(blocks)
    if not np.isfinite(features).all():
        raise ValueError('The encoder produced non-finite features.')
    return features, {'device':device,'torch':torch.__version__,'torchvision':torchvision.__version__,
                      'pillow':pillow_version,'encoder_dtype':str(model.dtype),
                      'encoder_checkpoint_sha256':expected_hash,
                      'rows':len(paths),'dimensions':features.shape[1],'batch_size':batch_size}


def save_features(out, rows, backbone, features, audit):
    dataset = rows.dataset.iloc[0]
    key = f'{SLUGS[dataset]}_{backbone}'
    ids = rows.sample_id.to_numpy(dtype=str)
    np.savez_compressed(out/f'{key}.npz',sample_ids=ids,features=features)
    digest = feature_fingerprint(ids,features)
    expected = json.loads((ROOT/'configs/feature_fingerprints.json').read_text())[key]
    write_json(out/f'{key}_extraction.json',dict(audit,dataset=dataset,rows=len(ids),
               sha256=digest,archived_feature_fingerprint_matches=digest==expected['sha256']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=list(SLUGS),required=True)
    parser.add_argument('--backbone',choices=list(MODELS),default='RN50')
    parser.add_argument('--images',type=Path,required=True)
    parser.add_argument('--layout',choices=['flat','indexed','generator'])
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--out',type=Path,default=ROOT/'local_data/features')
    args = parser.parse_args()
    if args.dataset == 'AGIQA-3K-full' and args.backbone != 'RN50':
        parser.error('Full AGIQA was evaluated with RN50 only.')
    layout = args.layout or ('indexed' if args.dataset == 'AIGCIQA2023' else 'flat')
    if args.dataset != 'AIGCIQA2023' and layout != 'flat':
        parser.error('AGIQA uses the official flat filenames.')
    rows = split_ids(args.dataset)
    paths = image_paths(rows,args.images,layout)
    out = output_directory(args.out)
    features, audit = encode_images(paths,args.backbone,args.checkpoint,args.device,args.batch_size,out/'encoder_cache')
    save_features(out,rows,args.backbone,features,audit)
    if args.dataset == 'AGIQA-3K-full':
        subset = split_ids('AGIQA-3K')
        indices = {s:i for i,s in enumerate(rows.sample_id)}
        values = features[[indices[s] for s in subset.sample_id]]
        save_features(out,subset,args.backbone,values,dict(audit,reused_from_full_protocol=True))
    print('Saved local features. The extraction report records whether they match the archived fingerprint.')


if __name__ == '__main__':
    main()
