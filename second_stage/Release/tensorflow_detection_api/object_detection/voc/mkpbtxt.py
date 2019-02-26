

src = "categories.txt"
dist = "billboard_label_map.pbtxt"
reader = open(src, 'r')
writer = open(dist, 'wb')

line = reader.readline()
while line:
    line = line.strip('\n')
    # line = line[:-2]
    cols = line.split(':')
    if len(cols) != 2:
        break
    #item = b'item {'+b'\n  '+b'id: '+bytes(cols[0], encoding='utf-8')+b'\n  '+b'name: \''+bytes(cols[1], encoding='utf-8')+b'\'\n'+b'}'+b'\n'
    # item = bytes(item, encoding='utf-8')
    item = "item {\n  "+"id: "+cols[0]+"\n  "+"name: 'n"+cols[0]+"'\n}\n"
    item = bytes(item, encoding="utf-8")
    print(type(item))
    writer.write(item)
    writer.flush()
    line = reader.readline()

reader.close()
writer.close()
