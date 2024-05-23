#

# create a zip file for submission

import os
import sys
import zipfile

required_files = {
    "run_llama.py",
    "src/llama.py",
    "src/optimizer.py",
    "src/classifier.py",
    "src/rope.py",
    "output/generated-sentence-temp-0.txt",
    "output/generated-sentence-temp-1.txt",
    "output/sst-dev-prompting-output.txt",
    "output/sst-test-prompting-output.txt",
    "output/sst-dev-finetuning-output.txt",
    "output/sst-test-finetuning-output.txt",
    "output/cfimdb-dev-prompting-output.txt",
    "output/cfimdb-test-prompting-output.txt",
    "output/cfimdb-dev-finetuning-output.txt",
    "output/cfimdb-test-finetuning-output.txt",
}


def check_file(file: str, check_aid: str):
    target_prefix = None
    # --
    inside_files = set()
    with zipfile.ZipFile(file, "r") as zz:
        # --
        print(f"Read zipfile {file}:")
        zz.printdir()
        print("#--")
        # --
        for info in zz.infolist():
            if info.filename.startswith("_"):
                continue  # ignore these files
            if target_prefix is None:
                target_prefix, _ = info.filename.split("/", 1)
                target_prefix = target_prefix + "/"
            # --
            assert info.filename.startswith(
                target_prefix
            ), "There should only be one top-level dir (with your andrew id as the dir-name) inside the zip file."
            ff = info.filename[len(target_prefix) :]
            inside_files.add(ff)
        # --
    # --
    assert target_prefix.startswith(
        check_aid
    ), f"AndrewID mismatched: {target_prefix[:-1]} vs {check_aid}"
    print(
        f"Read zipfile {file}, please check that your andrew-id is: {target_prefix[:-1]}"
    )
    print(f"And it contains the following files: {sorted(list(inside_files))}")
    # --


def create_zip_file(path: str, aid: str, zz: zipfile.ZipFile):
    for root, dirs, files in os.walk(path):
        if ".git" in root or "__pycache__" in root:
            continue  # ignore some parts
        for file in files:
            if file.endswith(".zip"):
                continue
            ff = os.path.join(root, file)
            rpath = os.path.relpath(ff, path)
            zz.write(ff, os.path.join(".", aid, rpath))
            if rpath in required_files:
                required_files.remove(rpath)


def main(path: str, aid: str):
    aid = aid.strip()
    if os.path.isdir(path):
        with zipfile.ZipFile(f"{aid}.zip", "w") as zz:
            create_zip_file(path, aid, zz)
            assert len(required_files) == 0, breakpoint()
            print(f"Submission zip file created from DIR={path} for {aid}: {aid}.zip")
            check_file(f"{aid}.zip", aid)
    else:  # directly check
        check_file(path, aid)


if __name__ == "__main__":
    main(*sys.argv[1:])
