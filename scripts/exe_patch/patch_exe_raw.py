import os
import sys
import struct
import marshal
import zlib
import shutil

sys.path.insert(0, r'D:\未完成项目\SitGuard')

from PyInstaller.archive.readers import CArchiveReader, ZlibArchiveReader
from PyInstaller.archive.writers import ZlibArchiveWriter
from PyInstaller.compat import BYTECODE_MAGIC
from PyInstaller.building.utils import get_code_object, strip_paths_in_code

SRC_EXE = r'D:\未完成项目\SitGuard\originals\SitGuard_original.exe'
FIXED_DEVICES_PY = r'D:\未完成项目\SitGuard\sitguard\devices.py'
FIXED_INIT_PY = r'D:\未完成项目\SitGuard\scripts\exe_patch\__init___patched.py'
OUT_EXE = r'D:\未完成项目\SitGuard\dist\SitGuard_v4.exe'
WORK_DIR = r'D:\未完成项目\SitGuard\scripts\exe_patch'

PYZ_NAME = 'PYZ-00.pyz'
TARGET_MODULE = 'sitguard.devices'


def read_cookie(path):
    COOKIE_FORMAT = '!8sIIii64s'
    COOKIE_LENGTH = struct.calcsize(COOKIE_FORMAT)
    with open(path, 'rb') as f:
        f.seek(-COOKIE_LENGTH, os.SEEK_END)
        cookie = f.read(COOKIE_LENGTH)
    magic, archive_length, toc_offset, toc_length, pyvers, pylib_name = struct.unpack(COOKIE_FORMAT, cookie)
    return {
        'magic': magic,
        'archive_length': archive_length,
        'toc_offset': toc_offset,
        'toc_length': toc_length,
        'pyvers': pyvers,
        'pylib_name': pylib_name.decode('ascii').rstrip('\x00'),
    }


def serialize_toc(toc_entries):
    TOC_ENTRY_FORMAT = '!iIIIBB'
    TOC_ENTRY_LENGTH = struct.calcsize(TOC_ENTRY_FORMAT)
    serialized = []
    for data_offset, compressed_length, data_length, compress, typecode, name in toc_entries:
        name_bytes = name.encode('utf-8')
        name_length = len(name_bytes) + 1
        entry_length = TOC_ENTRY_LENGTH + name_length
        if entry_length % 16 != 0:
            padding = 16 - (entry_length % 16)
            name_length += padding
        serialized_entry = struct.pack(
            TOC_ENTRY_FORMAT + f'{name_length}s',
            TOC_ENTRY_LENGTH + name_length,
            data_offset,
            compressed_length,
            data_length,
            compress,
            ord(typecode),
            name_bytes,
        )
        serialized.append(serialized_entry)
    return b''.join(serialized)


def main():
    print(f'Opening source exe: {SRC_EXE}')
    reader = CArchiveReader(SRC_EXE)
    cookie = read_cookie(SRC_EXE)
    print(f'Archive start offset: {reader._start_offset}')
    print(f'Original archive length: {cookie["archive_length"]}')

    # Read bootloader
    with open(SRC_EXE, 'rb') as f:
        bootloader = f.read(reader._start_offset)
    print(f'Bootloader size: {len(bootloader)}')

    # Read all entry raw blobs
    entry_blobs = {}
    for name, info in reader.toc.items():
        offset, data_length, uncompressed_length, compression_flag, typecode = info
        with open(SRC_EXE, 'rb') as f:
            f.seek(reader._start_offset + offset)
            blob = f.read(data_length)
        entry_blobs[name] = {
            'blob': blob,
            'uncompressed_length': uncompressed_length,
            'compression_flag': compression_flag,
            'typecode': typecode,
        }

    # Build new PYZ
    print(f'Building new PYZ from {PYZ_NAME}')
    pyz_blob = entry_blobs[PYZ_NAME]['blob']
    # PYZ entry is stored uncompressed (compression_flag=0), so blob is the PYZ file
    pyz_path = os.path.join(WORK_DIR, 'original_pyz_raw.pyz')
    with open(pyz_path, 'wb') as f:
        f.write(pyz_blob)

    zr = ZlibArchiveReader(pyz_path)
    code_dict = {}
    pyz_entries = []
    for name, info in zr.toc.items():
        typecode, offset, length = info
        code = zr.extract(name)
        code_dict[name] = code
        # Reconstruct src_path so ZlibArchiveWriter infers the same PYZ typecode
        if typecode == 3:  # PYZ_ITEM_NSPKG
            src_path = '-'
        elif typecode == 1 or name.endswith('.__init__') or name == 'sitguard':  # PYZ_ITEM_PKG
            # Package entry: name is package name, code is __init__.py
            src_path = os.path.join(WORK_DIR, name.replace('.', os.sep), '__init__.py')
        else:  # PYZ_ITEM_MODULE
            src_path = os.path.join(WORK_DIR, name.replace('.', os.sep) + '.py')
        pyz_entries.append((name, src_path, 'PYMODULE'))

    # Compile fixed devices.py
    print(f'Compiling fixed module: {FIXED_DEVICES_PY}')
    fixed_code = get_code_object(TARGET_MODULE, FIXED_DEVICES_PY)
    fixed_code = strip_paths_in_code(fixed_code)
    code_dict[TARGET_MODULE] = fixed_code

    # Compile patched package __init__ (global cv2.VideoCapture DSHOW fix)
    print(f'Compiling patched __init__: {FIXED_INIT_PY}')
    init_code = get_code_object('sitguard', FIXED_INIT_PY)
    init_code = strip_paths_in_code(init_code)
    code_dict['sitguard'] = init_code

    new_pyz_path = os.path.join(WORK_DIR, 'patched_pyz_raw.pyz')
    ZlibArchiveWriter(new_pyz_path, pyz_entries, code_dict)
    with open(new_pyz_path, 'rb') as f:
        new_pyz_blob = f.read()
    print(f'New PYZ size: {len(new_pyz_blob)} bytes (original: {len(pyz_blob)})')

    # Update PYZ blob in entry_blobs
    entry_blobs[PYZ_NAME]['blob'] = new_pyz_blob
    entry_blobs[PYZ_NAME]['uncompressed_length'] = len(new_pyz_blob)
    entry_blobs[PYZ_NAME]['compression_flag'] = 0  # PYZ stays uncompressed at CArchive level

    # Build new CArchive
    print('Building new CArchive...')
    new_toc_entries = []
    archive_data = bytearray()

    for name, info in reader.toc.items():
        blob = entry_blobs[name]['blob']
        offset = len(archive_data)
        archive_data.extend(blob)
        new_toc_entries.append((
            offset,
            len(blob),
            entry_blobs[name]['uncompressed_length'],
            entry_blobs[name]['compression_flag'],
            entry_blobs[name]['typecode'],
            name,
        ))

    toc_data = serialize_toc(new_toc_entries)
    toc_offset = len(archive_data)
    archive_data.extend(toc_data)

    # Write cookie
    COOKIE_FORMAT = '!8sIIii64s'
    COOKIE_MAGIC_PATTERN = b'MEI\014\013\012\013\016'
    archive_length = toc_offset + len(toc_data) + struct.calcsize(COOKIE_FORMAT)
    cookie_data = struct.pack(
        COOKIE_FORMAT,
        COOKIE_MAGIC_PATTERN,
        archive_length,
        toc_offset,
        len(toc_data),
        cookie['pyvers'],
        cookie['pylib_name'].encode('ascii'),
    )
    archive_data.extend(cookie_data)

    # Write final exe
    print(f'Writing final exe: {OUT_EXE}')
    with open(OUT_EXE, 'wb') as f:
        f.write(bootloader)
        f.write(archive_data)

    print(f'Final exe size: {os.path.getsize(OUT_EXE)} bytes')
    print(f'New archive length: {archive_length}')

    # Verify
    print('Verifying patched exe...')
    verify = CArchiveReader(OUT_EXE)
    print(f'Verify entries: {len(verify.toc)}')
    verify_pyz = verify.open_embedded_archive(PYZ_NAME)
    code = verify_pyz.extract(TARGET_MODULE)
    print(f'Fixed module {TARGET_MODULE} code size: {len(code.co_code)}')

    # Check CAP_DSHOW in nested code
    def find_name(code_obj, target):
        if target in code_obj.co_names:
            return True
        for const in code_obj.co_consts:
            if isinstance(const, type(code_obj)) and find_name(const, target):
                return True
        return False
    print(f'devices has CAP_DSHOW: {find_name(code, "CAP_DSHOW")}')

    init_verify = verify_pyz.extract('sitguard')
    print(f'__init__ has _install_camera_backend_fix: {find_name(init_verify, "_install_camera_backend_fix")}')
    print(f'__init__ has CAP_DSHOW string: {find_name(init_verify, "VideoCapture")}')

    print('Done.')


if __name__ == '__main__':
    main()
