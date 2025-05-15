from subprocess import check_output, CalledProcessError
import re
import hashlib


def calculate_md5(string):
    md5_hash = hashlib.md5()
    md5_hash.update(string.encode("utf-8"))
    md5_sum = md5_hash.hexdigest()
    return md5_sum


def wc(filename):
    try:
        result = int(check_output(["wc", "-l", filename]).split()[0])
    except CalledProcessError as exc:
        result = exc.output
    return result


def extract_code(text):
    pattern = r"```(.*?)\n([\s\S]*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) > 0 and matches != None:
        language = matches[0][0]
        code = matches[0][1]
        return language, code.strip()  # If the language does not match, return ''.
    return "", text
