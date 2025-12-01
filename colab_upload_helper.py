# colab_upload_helper_with_zip.py
# Run in Google Colab

import os
import re
import json
import hashlib
import time
import zipfile
import shutil
from pathlib import Path
from typing import Optional, Tuple, List, Union
from google.colab import files
import requests
import zipfile
import urllib.request
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
prefix_pattern = re.compile(r"^(\d+)_")


INPUT_DIR = Path("input")
META_FILE = INPUT_DIR / ".meta.json"
_SUFF_RE = re.compile(r"^(?P<base>.+?)(?:_(?P<num>\d+))?$")

def _ensure_input_dir():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not META_FILE.exists():
        _write_meta({"files": [], "last": None})

def _read_meta() -> dict:
    if not META_FILE.exists():
        return {"files": [], "last": None}
    try:
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"files": [], "last": None}

def _write_meta(d: dict):
    META_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def _split_name_suffix(filename: str) -> Tuple[str, Optional[int], str]:
    p = Path(filename)
    stem = p.stem
    m = _SUFF_RE.match(stem)
    if not m:
        return stem, None, p.suffix
    base = m.group("base")
    num = m.group("num")
    return base, (int(num) if num is not None else None), p.suffix

def _existing_variants(base: str, ext: str) -> List[Path]:
    res = []
    for f in INPUT_DIR.iterdir():
        if not f.exists():
            continue
        # include files and directories: for directories ext=='' and base matches folder name base/_N logic
        if f.is_file():
            b, n, e = _split_name_suffix(f.name)
            if e == ext and b == base:
                res.append(f)
        elif f.is_dir():
            # for dir, consider ext == '' and base==dir name base/_N
            if ext == '':
                # split dir name like file stem
                dbase, dnum, dext = _split_name_suffix(f.name)
                if dbase == base:
                    res.append(f)
    return res

def _next_variant_name(base: str, ext: str) -> str:
    variants = _existing_variants(base, ext)
    if not variants:
        return f"{base}{ext}"
    max_n = -1
    has_plain = False
    for v in variants:
        name = v.name
        b, n, e = _split_name_suffix(name)
        if n is None:
            has_plain = True
        else:
            if n > max_n:
                max_n = n
    if not has_plain and max_n == -1:
        return f"{base}{ext}"
    next_n = max_n + 1
    if max_n == -1 and has_plain:
        next_n = 1
    return f"{base}_{next_n}{ext}"

def _register_file_in_meta(name: str, file_hash: str, kind: str):
    meta = _read_meta()
    now = time.time()
    meta_entry = {"name": name, "hash": file_hash, "time": now, "kind": kind}
    meta["files"].append(meta_entry)
    meta["last"] = name
    _write_meta(meta)

def _find_existing_same_hash_file(hash_val: str, base: str, ext: str) -> Optional[Path]:
    variants = _existing_variants(base, ext)
    for v in variants:
        if v.is_file():
            try:
                b = v.read_bytes()
            except Exception:
                continue
            if _sha256_bytes(b) == hash_val:
                return v
    return None

def _compute_folder_hash(folder: Path) -> str:
    """
    Compute deterministic hash for a folder: for all files (recursively), sorted by relative path,
    compute sha256 of each file and then combine path + file_hash into one rolling sha256.
    """
    items = []
    for p in folder.rglob('*'):
        if p.is_file():
            rel = str(p.relative_to(folder)).replace(os.sep, '/')
            file_hash = _sha256_bytes(p.read_bytes())
            items.append((rel, file_hash))
    items.sort()  # sort by path
    h = hashlib.sha256()
    for rel, fh in items:
        # combine: path\0hash\0
        h.update(rel.encode('utf-8') + b'\0' + fh.encode('utf-8') + b'\0')
    return h.hexdigest()

def _find_existing_same_hash_folder(hash_val: str, base: str) -> Optional[Path]:
    variants = _existing_variants(base, '')  # ext == '' for directories
    for v in variants:
        if v.is_dir():
            try:
                fh = _compute_folder_hash(v)
            except Exception:
                continue
            if fh == hash_val:
                return v
    return None

def _safe_extract_zip(zip_path: Path, target_dir: Path):
    """
    Safely extract zip into target_dir, preventing path traversal.
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        for member in z.namelist():
            # disallow absolute paths and upward navigation
            if os.path.isabs(member) or '..' in Path(member).parts:
                raise RuntimeError(f"Unsafe path in zip: {member}")
        z.extractall(target_dir)

def _handle_uploaded_file_bytes(filename: str, data: bytes) -> Path:
    """
    Process one uploaded file (not zip): same logic as before.
    Returns the path of the saved file.
    """
    file_hash = _sha256_bytes(data)
    base, num, ext = _split_name_suffix(filename)
    same = _find_existing_same_hash_file(file_hash, base, ext)
    if same is not None:
        if same.name == filename:
            target = INPUT_DIR / filename
            tmp = INPUT_DIR / (filename + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(target)
            _register_file_in_meta(target.name, file_hash, kind="file")
            return target
        else:
            target = INPUT_DIR / filename
            tmp = INPUT_DIR / (filename + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(target)
            _register_file_in_meta(target.name, file_hash, kind="file")
            return target
    else:
        target_path = INPUT_DIR / filename
        if target_path.exists():
            new_name = _next_variant_name(base, ext)
            target = INPUT_DIR / new_name
        else:
            target = target_path
        tmp = INPUT_DIR / (target.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(target)
        _register_file_in_meta(target.name, file_hash, kind="file")
        return target

def _handle_uploaded_zip_bytes(zipname: str, data: bytes) -> Path:
    """
    Process one uploaded zip: unpack into a temporary folder, compute hash,
    then apply the same logic (replace if identical, else create variant folder).
    Returns the path to the created folder.
    """
    # temporary names
    tmp_zip = INPUT_DIR / (zipname + ".upload.tmp.zip")
    tmp_zip.write_bytes(data)

    # base folder name without extension
    base_name = Path(zipname).stem
    # unpack into a temporary subfolder, for example input/.tmp_<base>_<timestamp> 
    tmp_extract = INPUT_DIR / (f".tmp_extract_{base_name}_{int(time.time()*1000)}")
    tmp_extract.mkdir(parents=True, exist_ok=False)

    try:
        _safe_extract_zip(tmp_zip, tmp_extract)
    except Exception as e:
        # cleanup
        try:
            shutil.rmtree(tmp_extract)
        except Exception:
            pass
        tmp_zip.unlink(missing_ok=True)
        raise

    # compute folder hash
    folder_hash = _compute_folder_hash(tmp_extract)

    # search for an existing folder with the same hash
    same = _find_existing_same_hash_folder(folder_hash, base_name)
    if same is not None:
        # if content matches but name may differ — сохраним под имя загруженного zip (перезапишем если имя совпадает)
        target_dir = INPUT_DIR / base_name
        if same.name == base_name:
            # overwrite existing folder (удаляем и переименуем tmp_extract)
            # to be atomic: создаём tmp target и затем заменяем
            tmp_target = INPUT_DIR / (f".tmp_folder_{base_name}_{int(time.time()*1000)}")
            tmp_extract.rename(tmp_target)
            # remove old and rename tmp_target to desired
            if target_dir.exists():
                shutil.rmtree(target_dir)
            tmp_target.rename(target_dir)
            # write meta — используем существующую hash
            _register_file_in_meta(target_dir.name, folder_hash, kind="folder")
            tmp_zip.unlink(missing_ok=True)
            return target_dir
        else:
            # found same content under a different name, но пользователь загрузил zip с именем base_name.zip
            # just save unpacked content under zip name (создаём/перезаписываем base_name)
            target_dir = INPUT_DIR / base_name
            if target_dir.exists():
                shutil.rmtree(target_dir)
            tmp_extract.rename(target_dir)
            _register_file_in_meta(target_dir.name, folder_hash, kind="folder")
            tmp_zip.unlink(missing_ok=True)
            return target_dir
    else:
        # no content match: если имя уже занято -> выбрать variant name
        target_dir = INPUT_DIR / base_name
        if target_dir.exists():
            new_name = Path(_next_variant_name(base_name, ''))  # ext '' for dir
            target_dir = INPUT_DIR / new_name.name
        # rename temporary folder to target
        tmp_extract.rename(target_dir)
        _register_file_in_meta(target_dir.name, folder_hash, kind="folder")
        tmp_zip.unlink(missing_ok=True)
        return target_dir


def _download_pdb_to_bytes(pdb_code: str) -> bytes:
    """
    Download https://files.rcsb.org/view/{pdb_code}.pdb and return bytes.
    Use requests if available, otherwise urllib.
    Raises exception on error.
    """
    url = f"https://files.rcsb.org/view/{pdb_code}.pdb"
    # Try requests first
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.content
    except Exception:
        # fallback to urllib
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return r.read()
        except Exception as e:
            raise RuntimeError(f"Failed to download PDB {pdb_code} from {url}: {e}")

def process_upload(is_same: bool=False, pdb_code: Optional[str]=None) -> Optional[Union[Path, List[Path]]]:
    """
    Unified process:
      - If pdb_code provided and non-empty: download PDB at https://files.rcsb.org/view/{pdb_code}.pdb
        and process it as a single uploaded file (no upload dialog).
      - Else if is_same==True: return last saved entry from meta (without opening upload dialog).
      - Otherwise: open upload dialog (files.upload()) and process uploaded files (zip or single files).
    Returns Path or list of Paths (or None).
    """
    _ensure_input_dir()

    # 1) If pdb_code provided -> download and process directly (NO upload dialog)
    if pdb_code:
        pdb_code_str = str(pdb_code).strip()
        if pdb_code_str == "":
            print("Empty pdb_code provided — ignoring and proceeding normally.")
        else:
            print(f"Downloading PDB {pdb_code_str} ...")
            try:
                data = _download_pdb_to_bytes(pdb_code_str)
            except Exception as e:
                print("Error downloading PDB:", e)
                return None
            # Construct filename like '1abc.pdb' or preserve case: use pdb_code as provided
            filename = f"{pdb_code_str}.pdb"
            try:
                # Process as normal uploaded file (no root copy to clean, since we didn't use files.upload)
                p = _handle_uploaded_file_bytes(filename, data)
                return p
            except Exception as e:
                print("Error processing downloaded PDB file:", e)
                return None

    # 2) If user requested to use last -> return last immediately without opening upload dialog
    if is_same:
        meta = _read_meta()
        last = meta.get("last")
        if last:
            last_path = INPUT_DIR / last
            if last_path.exists():
                print(f"Using last (no upload dialog): {last_path}")
                return last_path
            else:
                print("Meta refers to last file but it doesn't exist on disk. Will proceed to upload dialog.")
        else:
            print("No previous file recorded. Will proceed to upload dialog.")

    # 3) Otherwise open upload dialog
    print("Upload file(s) (or zip). Close the dialog to skip.")
    uploaded = files.upload()  # dict: filename -> bytes

    # If user closed dialog and nothing uploaded
    if not uploaded:
        if is_same:
            # we already tried to return last above; if we get here, nothing uploaded and last missing
            print("No previously uploaded files and nothing uploaded now.")
            return None
        else:
            print("Files not uploaded.")
            return None

    saved_paths: List[Path] = []
    # Process each uploaded file
    for filename, data in uploaded.items():
        # Ensure bytes
        b = data if isinstance(data, (bytes, bytearray)) else data.read()

        # If files.upload() has also created a file in cwd, remember to remove it afterwards.
        root_copy = Path(filename)
        try:
            if filename.lower().endswith('.zip'):
                print(f"Processing zip: {filename}")
                p = _handle_uploaded_zip_bytes(filename, b)
                saved_paths.append(p)
            else:
                print(f"Processing file: {filename}")
                p = _handle_uploaded_file_bytes(filename, b)
                saved_paths.append(p)
        finally:
            # remove the copy that files.upload() may have written into current working dir
            try:
                if root_copy.exists():
                    root_copy.unlink()
            except Exception:
                # not critical, just logging
                print(f"Warning: couldn't remove root copy {root_copy}")

    if len(saved_paths) == 1:
        return saved_paths[0]
    return saved_paths

def create_output_folder(original_name: str) -> Path:
    """
    Create in output/ new dir with name:
       <N>_<original_name>
    where N — next available integer number.
    """
    max_n = 1 
    
    for p in OUTPUT_DIR.iterdir():
        if p.is_dir():
            m = prefix_pattern.match(p.name)
            if m:
                try:
                    num = int(m.group(1))
                    if num >= max_n:
                        max_n = num + 1
                except ValueError:
                    pass

    # New dir name
    new_name = f"{max_n}_{original_name}"
    new_path = OUTPUT_DIR / new_name

    new_path.mkdir(parents=True, exist_ok=False)
    return new_path

def archive_latest_output() -> Path:
    """
    Find the folder in output/ with last experiment
    and zip it.
    return path to arhive.
    """

    # find dir_s with prefix N_
    candidates = []
    for p in OUTPUT_DIR.iterdir():
        if p.is_dir():
            m = prefix_pattern.match(p.name)
            if m:
                try:
                    num = int(m.group(1))
                    candidates.append((num, p))
                except ValueError:
                    pass

    if not candidates:
        raise RuntimeError("In output/ there is no folders with prefix N_")

    
    candidates.sort(key=lambda x: x[0])
    max_num, latest_folder = candidates[-1]

    zip_path = OUTPUT_DIR / f"{latest_folder.name}.zip"

    # Create ZIP
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in latest_folder.rglob("*"):
            zf.write(
                file,
                arcname=file.relative_to(latest_folder)  # без полного пути
            )

    return zip_path