import click
import os

import xml.etree.ElementTree as ET


def get_pairs(tree):
    out = []
    for node in tree.iter(tag = "Name"):
        src = node.find("SourceName")
        tgts = node.findall("TargetName")
        for tgt in tgts:
            out.append((src.text, tgt.text))

    return out


@click.command()
@click.argument("dirname")
@click.argument("outdir")
def main(dirname, outdir):
    print(dirname, outdir)
    train, dev = None, None
    for fn in os.listdir(dirname):
        if fn.startswith("train"):
            traintree = ET.parse(os.path.join(dirname, fn))
            train = get_pairs(traintree)
        elif fn.startswith("dev"):
            devtree = ET.parse(os.path.join(dirname, fn))
            dev = get_pairs(devtree)

    if not train and not dev:
        msg = f"{dirname} does not have both a train and dev XML."
        raise Exception(msg)
    # Write parsed.
    dataset = os.path.basename(dirname)
    os.makedirs(os.path.join(outdir, dataset), exist_ok=True)
    with open(os.path.join(outdir, dataset, "train.tsv"), "w") as tout:
        for src, tgt in train:
            print(src, tgt, sep="\t", file=tout)
    with open(os.path.join(outdir, dataset, "dev.tsv"), "w") as dout:
        for src, tgt in dev:
            print(src, tgt, sep="\t", file=dout)

if __name__ == "__main__":
    main()