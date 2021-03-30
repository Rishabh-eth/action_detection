from fvcore.common.file_io import PathManager
import os


def read_labelmap(labelmap_file):
    """Read label map and class ids."""

    labelmap = []
    class_ids = set()
    name = ""
    class_id = ""
    with PathManager.open(labelmap_file, "r") as f:
        for line in f:
            if line.startswith("  name:"):
                name = line.split('"')[1]
            elif line.startswith("  id:") or line.startswith("  label_id:"):
                class_id = int(line.strip().split(" ")[-1])
                labelmap.append({"id": class_id, "name": name})
                class_ids.add(class_id)
    return labelmap, class_ids

def reduce_pbtxt_write_json(orig_annot, annot_dir, classes):
    # adapt classes such that it is indexed from 0 ...
    classes = [x - 1 for x in classes]

    labelmap, _ = read_labelmap(os.path.join(orig_annot, 'ava_action_list_v2.2.pbtxt'))

    # create pbtxt
    os.mknod(os.path.join(annot_dir, 'ava_action_list_v2.2.pbtxt'))
    new_label = 1
    with open(os.path.join(annot_dir, 'ava_action_list_v2.2.pbtxt'), 'a') as the_file:
        for cl in classes:
            the_file.write('label ')
            the_file.write('{\n')
            the_file.write('  name: ' + '"' + labelmap[cl]['name'] + '"')
            the_file.write('\n')
            # use new indexing of labels in order to have consecutive labels in the code
            the_file.write('  label_id: ' + str(new_label))
            #the_file.write('  label_id: ' + str(labelmap[cl]['id']))
            new_label += 1
            the_file.write('\n}\n')

    # create json
    os.mknod(os.path.join(annot_dir, 'class_names.json'))
    new_label = 0
    with open(os.path.join(annot_dir, 'class_names.json'), 'a') as the_file:
        the_file.write('{')
        for cl in classes:
            the_file.write('"' + labelmap[cl]['name'] + '"')
            the_file.write(': ')
            the_file.write(str(new_label))

            if new_label != len(classes)-1:
                the_file.write(', ')

            new_label += 1

        the_file.write('}')

