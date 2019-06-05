# pedxing-seq1-annot.idl.txt

f1 = "annotations/pedxing-seq1-annot.idl.txt"
f2 = "annotations/pedxing-seq2-annot.idl.txt"
f3 = "annotations/pedxing-seq3-annot.idl.txt"

def extract(seq):
    if seq == 1:
        filename = f1
    elif seq == 2:
        filename = f2
    else:
        filename = f3
    bbox = {}
    with open(filename, 'r') as f:
        content = f.read()
        frames= content.split(';')
        for frame in frames:
            img, boxes = frame.split(':')
            img = img.strip()
            img = img.strip('"')
            boxes = list(eval(boxes))
            bbox[img] = boxes
    return bbox

bbox1 = extract(1)
print(bbox1['pedxing-seq1/00001600.jpg'])