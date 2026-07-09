import argparse
import json
from pathlib import Path


def split_stream(
    input_path: str,
    out_mfsf: str,
    out_mfmf: str,
) -> dict:
    Path(out_mfsf).parent.mkdir(parents=True, exist_ok=True)
    Path(out_mfmf).parent.mkdir(parents=True, exist_ok=True)

    n = n_mfsf = n_mfmf = 0
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(out_mfsf, 'w', encoding='utf-8') as f_mfsf, \
         open(out_mfmf, 'w', encoding='utf-8') as f_mfmf:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            n += 1

            is_mf = bool(obj.get('is_multi_function'))
            is_mfi = bool(obj.get('is_multi_file'))

            if is_mf and not is_mfi:
                f_mfsf.write(json.dumps(obj, ensure_ascii=False) + '\n')
                n_mfsf += 1
            elif is_mf and is_mfi:
                f_mfmf.write(json.dumps(obj, ensure_ascii=False) + '\n')
                n_mfmf += 1

    return {
        'total': n,
        'mfsf': n_mfsf,
        'mfmf': n_mfmf,
    }


def main():
    p = argparse.ArgumentParser(description='Split MF dataset into Multi-Function-Single-File (MFSF) and Multi-Function-Multi-File (MFMF).')
    p.add_argument('-i', '--input', required=True, help='Path to FuncFileRepo-MF.jsonl')
    p.add_argument('--mfsf', default='data/FuncFileRepo-MFSF.jsonl', help='Output path for MFSF JSONL')
    p.add_argument('--mfmf', default='data/FuncFileRepo-MFMF.jsonl', help='Output path for MFMF JSONL')
    args = p.parse_args()

    stats = split_stream(args.input, args.mfsf, args.mfmf)
    print(stats)


if __name__ == '__main__':
    main()
