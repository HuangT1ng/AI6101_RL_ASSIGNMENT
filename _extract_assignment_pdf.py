"""Small local extractor for this PDF's encoded text streams."""
import pathlib
import re
import zlib

pdf = pathlib.Path('2026Fall_AI6101_RL_Individual Assignment.pdf').read_bytes()
matches = list(re.finditer(rb'(?m)^(\d+) 0 obj\b', pdf))
objects = {}
for i, match in enumerate(matches):
    end = matches[i + 1].start() if i + 1 < len(matches) else len(pdf)
    objects[int(match.group(1))] = pdf[match.end():end]


def stream(number):
    obj = objects[number]
    match = re.search(rb'stream\r?\n(.*?)\r?\nendstream', obj, re.S)
    data = match.group(1)
    return zlib.decompress(data) if b'/FlateDecode' in obj[:match.start()] else data


def unescape(raw):
    result = bytearray()
    i = 0
    special = {ord(b'n'): 10, ord(b'r'): 13, ord(b't'): 9,
               ord(b'b'): 8, ord(b'f'): 12}
    while i < len(raw):
        c = raw[i]
        if c != 92 or i + 1 == len(raw):
            result.append(c)
            i += 1
            continue
        i += 1
        c = raw[i]
        if 48 <= c <= 55:
            digits = raw[i:i + 3]
            length = len(re.match(rb'[0-7]{1,3}', digits).group())
            result.append(int(raw[i:i + length], 8))
            i += length
        elif c in (10, 13):
            i += 2 if c == 13 and raw[i + 1:i + 2] == b'\n' else 1
        else:
            result.append(special.get(c, c))
            i += 1
    return bytes(result)


cmaps = {}
for number, obj in objects.items():
    if b'/ToUnicode' not in obj or b'/Type /Font' not in obj:
        continue
    cmap_number = int(re.search(rb'/ToUnicode (\d+) 0 R', obj).group(1))
    cmap_data = stream(cmap_number)
    mapping = {}
    for start, end, target in re.findall(rb'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', cmap_data):
        start, end, target = int(start, 16), int(end, 16), int(target, 16)
        for code in range(start, end + 1):
            mapping[code] = chr(target + code - start)
    cmaps[number] = mapping

string_re = re.compile(rb'\((?:\\.|[^\\)])*\)')
for page_no, (content_number, resource_number) in enumerate([(3, 4), (33, 34), (41, 42)], 1):
    resources = objects[resource_number]
    fonts = {name.decode(): cmaps.get(int(num), {}) for name, num in
             re.findall(rb'/(TT\d+)\s+(\d+) 0 R', resources)}
    fragments = []
    for block in re.findall(rb'BT(.*?)ET', stream(content_number), re.S):
        font_match = re.search(rb'/(TT\d+)\s+[\d.]+\s+Tf', block)
        pos_match = re.search(rb'([-\d.]+)\s+([-\d.]+)\s+Tm', block)
        if not (font_match and pos_match):
            continue
        font = fonts.get(font_match.group(1).decode(), {})
        source = b''.join(unescape(value[1:-1]) for value in string_re.findall(block))
        decoded = ''.join(font.get(c, '?') for c in source)
        if decoded.strip():
            x, y = (float(pos_match.group(1)), float(pos_match.group(2)))
            fragments.append((round(y / 8) * 8, x, decoded))
    rows = {}
    for y, x, value in fragments:
        rows.setdefault(y, []).append((x, value))
    print(f'\n===== PAGE {page_no} =====')
    for y in sorted(rows, reverse=True):
        print(' '.join(value for _, value in sorted(rows[y])))
