import json
import io
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

def writeJson(json, path):
    try:
        with open(path, 'w') as outfile:
            print(outfile)
            str_ = json.dumps(json, indent=4, sort_keys=True,
                              separators=(',', ': '), ensure_ascii=False)
            outfile.write(to_unicode(str_))
            #json.dump(path, outfile)
            return True, None
    except Exception as err:
        return False, err


def readJson(path):
    try:
        with open(path) as json_file:
            data = json.load(json_file)
            return True, data
    except Exception as err:
        False, err
