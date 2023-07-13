import re
import logging
import pandas as pd
from constants import mask_token

# Allows individual control of logging
logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def convert_into_triple_lists(offset_list):
    triples = []
    for ol in offset_list:
        match = re.match("^[ ]*{'end_offset': ([0-9]+), 'start_offset': ([0-9]+), 'word': ['\"]([^ ]+)['\"]}[ ]*$", ol)
        assert match
        triples.append([int(match.group(1)), int(match.group(2)), match.group(3)])
    assert len(triples) == len(offset_list)
    return triples


def exact_times(list1, list2):    
    # Compare just times
    times1 = [[x[0], x[1]] for x in list1]
    times2 = [[x[0], x[1]] for x in list2]

    # Find exact timing matches
    indexes1 = [0]
    indexes2 = [0]
    for i in range(len(times1)):
        if times1[i] in times2:
            indexes1.append(i)
            indexes2.append(times2.index(times1[i]))

    return indexes1, indexes2
    

def in_proximity(a, b, dist):
    if abs(a-b) <= dist:
        return True
    return False


def reconstruct_original(asr_list, asr_words):
    originals = []
    for i in asr_list:
        originals.append(asr_words[i][2])
    return originals
    
    
def start_end_align(list1, list2, tolerance_length, tolerance_offset):
    compare1 = [[x[0], x[1]] for x in list1]
    compare2 = [[x[0], x[1]] for x in list2]
    
    indexes1 = []
    indexes2 = []

    # Treat 0 position separately
    i = 0
    j = 0
    if in_proximity(compare1[i][1], compare2[j][1], tolerance_length):
        if in_proximity(compare1[i][0], compare2[j][0], tolerance_offset):
            indexes1.append(i)
            indexes2.append(j)

    if len(indexes1) > 0:
        assert len(indexes1) == len(indexes2)
        return indexes1, indexes2
    
    for i in range(1, len(compare1)):
        for j in range(i, len(compare2)):
            if in_proximity(compare1[i][1], compare2[j][1], tolerance_length):
                if in_proximity(compare1[i][0], compare2[j][0], tolerance_offset):
                    indexes1.append(i)
                    indexes2.append(j)
                    break

    return indexes1, indexes2
            

def word_count_align(list1, list2):
    # Does not take positions into account, just uses word counts
    if len(list1) == len(list2):
        indexes1 = list(range(len(list1)))
        indexes2 = list(range(len(list2)))
    else:
        indexes1 = []
        indexes2 = []
        
    return indexes1, indexes2
        

# position: 0 for end, 1 for start offsets
def close_align(position, list1, list2, tolerance_length, tolerance_offset):
    if (len(list1) == 2 and len(list2) == 1) or (len(list2) == 2 and len(list1) == 1):
        return [], []
    
    compare1 = [[x[position], (x[0] - x[1])] for x in list1]
    compare2 = [[x[position], (x[0] - x[1])] for x in list2]

    flip = False
    if len(compare1) > len(compare2):
        flip = True
        temp_compare = compare1
        compare1 = compare2
        compare2 = temp_compare
    
    indexes1 = []
    indexes2 = []
    add_boundary = False

    # Check 0 position separately
    i = 0
    j = 0
    if in_proximity(compare1[i][1], compare2[j][1], tolerance_length):
        if in_proximity(compare1[i][0], compare2[j][0], tolerance_offset):
            indexes1.append(i)
            indexes2.append(j)
            # Adding boundary (must be new)
            if (i + 1) < len(compare1):
                indexes1.append(i+1)
                indexes2.append(j+1)
                if flip:
                    return indexes2, indexes1
                else:
                    return indexes1, indexes2
            
    # Now there's opportunity for merging, so don't have to be 1-2-1
    for i in range(1, len(compare1)):
        for j in range(i,len(compare2)):
            if in_proximity(compare1[i][1], compare2[j][1], tolerance_length):
                if in_proximity(compare1[i][0], compare2[j][0], tolerance_offset):
                    if i in indexes1:
                        assert j in indexes2
                    else:
                        # If final element in 1 but not 2, can't add
                        if i == (len(compare1) - 1) and j != (len(compare2) - 1):
                            continue
                        
                        indexes1.append(i)
                        indexes2.append(j)
                    # Adding boundary (must be new)
                    if (i + 1) < len(compare1):
                        indexes1.append(i+1)
                        indexes2.append(j+1)
                    if len(indexes1) > 0:
                        if flip:
                            return indexes2, indexes1
                        else:
                            return indexes1, indexes2

    if flip:
        return indexes2, indexes1
    else:
        return indexes1, indexes2
    

def merge_align(position, list1, list2, tolerance_length, tolerance_offset):

    # Do not carry out merge align if the two lists are the same lengths
    if len(list1) == len(list2):
        return list(range(len(list1))), list(range(len(list2)))

    # If list1 has just one element, then need to return list2
    if len(list1) == 1:
        indexes1 = list(range(len(list1)))
        indexes2 = [list(range(len(list2)))]
        return indexes1, indexes2

    # And the other way around
    if len(list2) == 1:
        indexes1 = [list(range(len(list1)))]
        indexes2 = list(range(len(list2)))
        return indexes1, indexes2
    
    compare1 = [[x[position], (x[0] - x[1])] for x in list1]
    compare2 = [[x[position], (x[0] - x[1])] for x in list2]

    indexes1 = []
    indexes2 = []

    add_boundary = False

    # Want to compare [0] position first
    i = 0
    j = 0
    if in_proximity(compare1[i][0], compare2[j][0], tolerance_offset):
        total = 0
        onwards = j
        moved_end = compare2[j][1]
        while not in_proximity(compare1[i][1], moved_end, tolerance_length):
            if compare1[i][1] < moved_end:
                break
            total += compare2[onwards][1]
            if (onwards + 1) <= (len(compare2) - (len(compare1) - i)):
                onwards += 1
                moved_end = compare2[onwards][1] + total
            else:
                break
        if in_proximity(compare1[i][1], moved_end, tolerance_length):
            appendable = list(range(j, onwards+1))
            if len(appendable) > 1:
                indexes1.append(i)
                indexes2.append(appendable)
            else:
                indexes1.append(i)
                indexes2.append(appendable[0])
            indexes1.append(i + 1)
            assert isinstance(indexes2, list)
            indexes2.append(list(flatten(indexes2))[-1] + 1)

    assert len(indexes1) == len(indexes2)
    if len(indexes1) > 0:
        # Ensure that next loop takes changes into account
        return indexes1, indexes2
        
    # If within tolerance, check lengths
    # Increment until length within tolerance or exceeded    
    for i in range(1, len(compare1)):
        for j in range(i,len(compare2)):
            if in_proximity(compare1[i][0], compare2[j][0], tolerance_offset):
                total = 0
                onwards = j
                moved_end = compare2[j][1]
                while not in_proximity(compare1[i][1], moved_end, tolerance_length) or i == onwards:
                    if compare1[i][1] < moved_end:
                        break
                    total += compare2[onwards][1]
                    if (onwards + 1) <= (len(compare2) - (len(compare1) - i)):
                        onwards += 1
                        moved_end = compare2[onwards][1] + total
                    else:
                        break
                if in_proximity(compare1[i][1], moved_end, tolerance_length):
                    appendable = list(range(j, onwards+1))
                    if i == (len(compare1) - 1):
                        appendable = list(range(j, len(compare2)))
                    if len(appendable) > 1:
                        indexes1.append(i)
                        indexes2.append(appendable)
                    else:
                        indexes1.append(i)
                        indexes2.append(appendable[0])
                    assert len(indexes1) == 1
                    indexes1 = [indexes1] + [i + 1]
                    assert isinstance(indexes2, list)
                    indexes2.append(list(flatten(indexes2))[-1] + 1)
                    assert len(indexes1) == len(indexes2)
                    if len(indexes1) > 0:
                        # Ensure that next loop takes changes into account
                        return indexes1, indexes2
                    
    assert len(indexes1) == len(indexes2)
    return indexes1, indexes2
    

def flatten(L):
    for item in L:
        try:
            yield from flatten(item)
        except TypeError:
            yield item


def list_flatten(L):
    if isinstance(L, list):
        return list(flatten(L))
    else:
        return [L]


def approximate_alignment(indexes, length):
    # Compensating for poor alignments resulting in skipped NEs
    if indexes == list(range(length + 1)):
        return True
    elif list_flatten(indexes) == list(range(length + 1)):
        return True
    return False

    
def timed_align(list1, list2):

    tl = 0
    ts = 0
    te = 0
    position = 1

    indexes1 = [0, len(list1)]
    indexes2 = [0, len(list2)]

    # Due to adding boundaries, we'll not know if these have been done
    done_first = False
    done_last = True
    
    # Assume list2 is the shorter one
    while len(list(flatten(indexes2))) < len(list2) and not approximate_alignment(indexes1, len(list1)) and not approximate_alignment(indexes2, len(list2)):
        new_indexes1 = [indexes1[0]]
        new_indexes2 = [indexes2[0]]
        for i in range(len(indexes1) - 1):
            # If the (indexes1[i] + 1) == indexes1[i+1]
            if (list_flatten(indexes1[i])[-1] + 1) == list_flatten(indexes1[i+1])[0]:
                # No way to tell if first one's been accounted for
                if i > 0 or isinstance(indexes2[i], list):
                    # No intervening missing entries
                    if not indexes1[i] in new_indexes1:
                        new_indexes1.append(indexes1[i])
                    if not indexes2[i] in new_indexes2:
                        new_indexes2.append(indexes2[i])
                    # Final indexes
                    if not indexes1[i+1] in new_indexes1:
                        new_indexes1.append(indexes1[i + 1])
                    if not indexes2[i+1] in new_indexes2:
                        new_indexes2.append(indexes2[i + 1])           
                    continue

            # Some still missing
            if (list_flatten(indexes1[i])[-1] + 1) < list_flatten(indexes1[i+1])[0]:
                # No way to tell if first one's been accounted for
                if i > 0 and isinstance(indexes2[i], list):
                    # No intervening missing entries
                    if not indexes1[i] in new_indexes1:
                        new_indexes1.append(indexes1[i])
                    if not indexes2[i] in new_indexes2:
                        new_indexes2.append(indexes2[i])
                    # Final indexes
                    final_index1 = indexes1[i] + 1
                    if not final_index1 in new_indexes1:
                        new_indexes1.append(final_index1)
                    final_index2 = indexes2[i][-1] + 1
                    if not final_index2 in new_indexes2:
                        new_indexes2.append(final_index2)
                    continue
                
            if position == 4:
                # Run comparison based on numbers of words identified
                i1, i2 = word_count_align(list1[list_flatten(indexes1[i])[-1]:list_flatten(indexes1[i+1])[0]], list2[list_flatten(indexes2[i])[-1]:list_flatten(indexes2[i + 1])[0]])
            elif position == 2:
                # Run comparison based on start and end
                i1, i2 = start_end_align(list1[list_flatten(indexes1[i])[-1]:list_flatten(indexes1[i+1])[0]], list2[list_flatten(indexes2[i])[-1]:list_flatten(indexes2[i + 1])[0]], tl, ts)
            elif position == 3:
                # Run merge attempts as nothing else worked at this tl & ts
                i1, i2 = merge_align(1, list1[list_flatten(indexes1[i])[-1]:list_flatten(indexes1[i+1])[0]], list2[list_flatten(indexes2[i])[-1]:list_flatten(indexes2[i + 1])[0]], tl, ts)
            else:
                # Start *or* end align
                i1, i2 = close_align(position, list1[list_flatten(indexes1[i])[-1]:list_flatten(indexes1[i+1])[0]], list2[list_flatten(indexes2[i])[-1]:list_flatten(indexes2[i + 1])[0]], tl, ts)

            # Add indexes in between steps, taking offsets into account
            for entry in i1:

                if isinstance(entry, list):
                    if len(entry) == 1:
                        entry = entry[0]                        
                
                if isinstance(entry, list):
                    shift = [x + indexes1[i] for x in entry]
                else:
                    shift = entry + indexes1[i]
                end = indexes1[i+1]
                if shift in new_indexes1:
                    # Start may be repeated
                    continue
                elif isinstance(shift, list):
                    # Elements of shift may already be in the list. These
                    # should be removed and replaced.
                    for value in shift:
                        if value in new_indexes1:
                            new_indexes1.remove(value)
                    
                if end in new_indexes1:
                    # Make sure this is entered in the correct location
                    new_indexes1.insert(new_indexes1.index(end), shift)
                else:
                    new_indexes1.append(shift)

            # Need to also insert the next element along if that's
            # not the final one, to ensure this entry is avoided in
            # the future.
            for entry in i2:

                if isinstance(entry, list):
                    if len(entry) == 1:
                        entry = entry[0]
                
                if isinstance(entry, list):
                    if isinstance(indexes2[i], list):
                        shift = entry + indexes2[i]
                    else:
                        shift = [x + indexes2[i] for x in entry]
                else:
                    if isinstance(indexes2[i], list):
                        shift = [entry] + indexes2[i]
                    else:
                        shift = entry + indexes2[i]
                end = indexes2[i+1]
                if shift in new_indexes2:
                    continue
                elif isinstance(shift, list):
                    # Elements of shift may already be in the list. These
                    # should be removed and replaced.
                    for value in shift:
                        if value in new_indexes2:
                            new_indexes2.remove(value)
                if end in new_indexes2:
                    new_indexes2.insert(new_indexes2.index(end), shift)
                else:
                    new_indexes2.append(shift)
            # Previous iteration's final indexes
            if indexes1[i+1] in new_indexes1:
                # This was added as a boundary within a match
                assert indexes2[i+1] in new_indexes2
            else:
                if not indexes1[i + 1] in new_indexes1:
                    new_indexes1.append(indexes1[i + 1])
                if not indexes2[i + 1] in new_indexes2:
                    new_indexes2.append(indexes2[i + 1])
            assert len(new_indexes1) == len(new_indexes2)
        if position == 4:
            if new_indexes1 == indexes1 and new_indexes2 == indexes2:
                # Compare starts
                position = 1
        elif position == 1:
            if new_indexes1 == indexes1 and new_indexes2 == indexes2:
                # Compare ends
                position = 0
        elif position == 0:
            if new_indexes1 == indexes1 or new_indexes2 == indexes2:
                # Run start end aligns
                position = 2
        elif position == 2:
            if new_indexes1 == indexes1 or new_indexes2 == indexes2:
                # Run merge aligns
                position = 3
        else:
            # Must have completed merge aligns
            assert position == 3
            if new_indexes1 == indexes1 or new_indexes2 == indexes2:
                # Increase tolerances
                tl += 1
                ts += 1
                logger.debug('Increasing tolerance to: %d %d' % (tl, ts))
                position = 4
        assert len(new_indexes1) == len(new_indexes2)
        indexes1 = new_indexes1
        indexes2 = new_indexes2

    logger.debug('Timed align is returning: %s vs %s' % (indexes1, indexes2))
    assert len(indexes1) == len(indexes2)
    return indexes1[:-1], indexes2[:-1]

        
def check_if_contains_ne(indexes, words):
    contains_ne = False
    for index in indexes:
        if words[index][2].startswith('n'):
            contains_ne = True
            break
    return contains_ne


def construct_masked_sent(asr_words, ne_words, asr_align, ne_align, masking_model):
    sentence = ""
    originals = []
    for i in range(len(asr_align)):
        if isinstance(asr_align[i], list):
            if ne_words[ne_align[i]][2].startswith('n'):
                if sentence != "":
                    sentence += " "
                sentence += mask_token[masking_model]
                if asr_words[asr_align[i][-1]][2].endswith("'s"):
                    sentence += "'s"
                originals.append([reconstruct_original(asr_align[i], asr_words)])
            else:
                for index in asr_align[i]:
                    if sentence != "":
                        sentence += " "
                    sentence += asr_words[index][2]              
        elif isinstance(ne_align[i], list):
            # Go with NE if any of the entries is an NE entry
            if sentence != "":
                sentence += " "
            if check_if_contains_ne(ne_align[i], ne_words):
                originals.append(asr_words[asr_align[i]][2])
                sentence += mask_token[masking_model]
                if asr_words[asr_align[i]][2].endswith("'s"):
                    sentence += "'s"
            else:
                sentence += asr_words[asr_align[i]][2]
        else:
            # One to one mapping
            if sentence != "":
                sentence += " "
            if ne_words[ne_align[i]][2].startswith('n'):
                sentence += mask_token[masking_model]
                if asr_words[asr_align[i]][2].endswith("'s"):
                    sentence += "'s"
                originals.append(asr_words[asr_align[i]][2])
            else:
                sentence += asr_words[asr_align[i]][2]
    return " ".join([x[2] for x in asr_words]), sentence, originals
    

def align_and_mask(row, masking_model):
    ne_offset_list = row['ne_offsets'][1:-1].split('\n')
    asr_offset_list = row['asr_offsets'][1:-1].split('\n')

    # Convert into a list of predicted triples
    ne_pred_lists = convert_into_triple_lists(ne_offset_list)
    asr_pred_lists = convert_into_triple_lists(asr_offset_list)

    # Convert into a list of lengths
    ne_length_lists = [len(w) for w in ne_pred_lists]
    asr_length_lists = [len(w) for w in asr_pred_lists]

    if len(asr_length_lists) < len(ne_length_lists):
        asr_index, ne_index = timed_align(asr_pred_lists, ne_pred_lists)
        asr_sent, masked_sent, originals = construct_masked_sent(asr_pred_lists, ne_pred_lists, asr_index, ne_index, masking_model)
    else:
        ne_index, asr_index = timed_align(ne_pred_lists, asr_pred_lists)
        asr_sent, masked_sent, originals = construct_masked_sent(asr_pred_lists, ne_pred_lists, asr_index, ne_index, masking_model)

    return asr_sent, masked_sent, originals
    
