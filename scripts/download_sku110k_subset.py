import argparse
import io
import os
import tarfile

import requests

URL = "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300, help="numero de imagens a extrair")
    parser.add_argument("--skip", type=int, default=0, help="numero de imagens correspondentes a saltar antes de comecar a extrair")
    parser.add_argument("--out", type=str, default="data/images/raw_sku110k")
    parser.add_argument("--subset", type=str, default="train", help="train|test|val ou vazio para qualquer")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"A ligar a {URL} ...")
    resp = requests.get(URL, stream=True, timeout=60)
    resp.raise_for_status()

    fileobj = resp.raw
    fileobj.decode_content = True

    extracted = 0
    seen = 0
    with tarfile.open(fileobj=fileobj, mode="r|gz") as tar:
        for member in tar:
            if extracted >= args.n:
                break
            if not member.isfile():
                continue
            name_lower = member.name.lower()
            if not name_lower.endswith((".jpg", ".jpeg", ".png")):
                continue
            if args.subset and args.subset not in name_lower:
                continue

            seen += 1
            if seen <= args.skip:
                continue

            f = tar.extractfile(member)
            if f is None:
                continue
            data = f.read()

            out_name = os.path.basename(member.name)
            out_path = os.path.join(args.out, out_name)
            with open(out_path, "wb") as out_f:
                out_f.write(data)

            extracted += 1
            if extracted % 25 == 0:
                print(f"  {extracted}/{args.n} imagens extraidas")

    resp.close()
    print(f"Concluido: {extracted} imagens guardadas em {args.out}")


if __name__ == "__main__":
    main()
