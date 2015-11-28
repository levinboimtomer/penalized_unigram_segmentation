
def process_lines(original_lines, remove_whitespace, chop):
    if remove_whitespace:
        original_lines = [line.replace(' ', '') for line in original_lines]
    if chop > 0:
        original_lines = [line[:chop] for line in original_lines]
    return original_lines


def readlines(filename, rstrip=False):
    lines = []
    with open(filename, 'rb') as f:
        for (i, line) in enumerate(f):
            if rstrip:
                line = line.rstrip()
            lines.append(line)

    return lines

# counts the number of spaces in str that match, mismatch or are missing compared to the gold string
def compare_whitespace(str, gold):
    space_count_s = 0
    space_count_g = 0
    correct_space = 0
    incorrect_space = 0
    missed_space = 0

    s = 0
    g = 0
    while s < len(str) and g < len(gold):
        # count
        if str[s] == ' ':
            space_count_s += 1
        if gold[g] == ' ':
            space_count_g += 1

        # compare
        if str[s] == gold[g]:
            if str[s] == ' ':
                correct_space += 1
            s += 1
            g += 1
        elif str[s] == ' ':
            s += 1
            incorrect_space += 1
        elif gold[g] == ' ':
            g += 1
            missed_space += 1
        #print s, g

    # count spaces in the remaining parts
    gold_tail_missed_space = sum([x == ' ' for x in gold[g:]])
    str_tail_incorrect_space = sum([x == ' ' for x in str[s:]])

    missed_space += gold_tail_missed_space
    space_count_g += gold_tail_missed_space
    incorrect_space += str_tail_incorrect_space
    space_count_s += str_tail_incorrect_space

    assert space_count_s == correct_space + incorrect_space
    assert space_count_g == correct_space + missed_space

    return correct_space, incorrect_space, missed_space