
# coding: utf-8

import numpy as np

import config


def build_character_dictionary():

    characters_map = {}
    characters_count = 4
    with open('/mnt/ztf/seq2seq_for_nlp/data/cityu_training.utf8', 'r') as f:
        lines = f.readlines(-1)

        pass_count = 0
        # 0: padding, 1: eos, 2: sos, 3: unregisted
        process_count = 0
        for sentence in lines:
            if np.random.rand() > 0.88:
                pass_count += 1
                continue
            else:
                process_count += 1
            # sentence_index = []
            # words = sentence.split(config.seg)
            words = sentence.strip('\n').strip('\r').split(config.seg)
            for word in words:
                for decodec_character in word.decode('utf-8'):
                    character = decodec_character.encode('utf-8')
                    if characters_map.get(character) is not None:
                        # sentence_index.append(characters_map.get(character))
                        pass
                    else:
                        print('Handling character: %s' % character)
                        characters_map[character] = characters_count
                        characters_count += 1
        print('pass_count: %d' % pass_count)
        print('process_count: %d' % process_count)
        print('characters num: %d' % (characters_count))
        f.close()
    with open('/mnt/ztf/seq2seq_for_nlp/data/cityu_characters.txt', 'w') as f:
        for item in characters_map.items():
            f.writelines(['%s,%d\n' % (item[0], item[1])])
        f.close()


def build_sentences_set(sos_eos=True):

    characters_map = {}
    reversed_map = {}
    with open('/mnt/ztf/seq2seq_for_nlp/data/cityu_characters.txt', 'r') as f:
        lines = f.readlines(-1)
        for item in lines:
            kv = item.split(',')
            try:
                characters_map[kv[0]] = int(kv[1])
                reversed_map[int(kv[1])] = kv[0]
            except Exception as e:
                print('Sentence with error:')
                print(item)
        f.close()
    reversed_map[config.unregisted] = '[unregisted]'

    sentence_count = 0
    total_character_count = 0
    max_seq_len = config.max_seq_len - 2 if sos_eos is True else config.max_seq_len
    sentence_vectors = []
    labels = []
    with open('/mnt/ztf/seq2seq_for_nlp/data/cityu_training.utf8', 'r') as f:
        lines = f.readlines(-1)
        longest_seq_len = 0
        for sentence in lines:
            sentence_count += 1
            sentence_vector = []
            label = []
            words = sentence.strip('\n').strip('\r').split(config.seg)
            seq_len = 0
            for word in words:
                decodec_characters = word.decode('utf-8')
                seq_len += len(decodec_characters)
                for i in range(0, len(decodec_characters)):
                    decodec_character = decodec_characters[i]
                    character = decodec_character.encode('utf-8')
                    if characters_map.get(character) is not None:
                        sentence_vector.append(characters_map.get(character))
                    else:
                        # unregisted
                        sentence_vector.append(config.unregisted)
                    if len(decodec_characters) >= 2:
                        if i == 0:
                            tag = config.begin
                        elif i == len(decodec_characters) - 1:
                            tag = config.end
                        else:
                            tag = config.inner
                    else:
                        tag = config.single
                    label.append(tag)
            if seq_len > longest_seq_len:
                longest_seq_len = seq_len
            total_character_count += seq_len

            if seq_len <= max_seq_len:
                if sos_eos:
                    sentence_vector.insert(0, config.sos)
                    label.insert(0, config.sos)
                    sentence_vector.append(config.eos)
                    label.append(config.eos)
                sentence_vectors.append(np.asarray(sentence_vector))
                labels.append(np.asarray(label))
        f.close()
        print('Longest sentence length: %d' % longest_seq_len)
        print('Average sentence length: %.2f' % (float(total_character_count) / sentence_count))
    return sentence_vectors, labels, characters_map, reversed_map


def build_test_set(sos_eos=True):

    characters_map = {}
    reversed_map = {}
    with open('/mnt/ztf/seq2seq_for_nlp/data/cityu_characters.txt', 'r') as f:
        lines = f.readlines(-1)
        for item in lines:
            kv = item.split(',')
            try:
                characters_map[kv[0]] = int(kv[1])
                reversed_map[int(kv[1])] = kv[0]
            except Exception as e:
                print('Sentence with error:')
                print(item)
        f.close()
    reversed_map[config.unregisted] = '[unregisted]'

    sentence_count = 0
    total_character_count = 0
    max_seq_len = config.max_seq_len - 2 if sos_eos is True else config.max_seq_len
    sentence_vectors = []
    labels = []
    with open('/mnt/ztf/seq2seq_for_nlp/data/cityu_test.utf8', 'r') as f:
        lines = f.readlines(-1)
        longest_seq_len = 0
        for sentence in lines:
            sentence_count += 1
            sentence_vector = []
            label = []
            sentence = sentence.strip('\n').strip('\r')

            decodec_characters = sentence.decode('utf-8')
            for i in range(0, len(decodec_characters)):
                decodec_character = decodec_characters[i]
                character = decodec_character.encode('utf-8')
                if characters_map.get(character) is not None:
                    sentence_vector.append(characters_map.get(character))
                else:
                    # unregisted
                    sentence_vector.append(config.unregisted)
                tag = config.pad
                label.append(tag)

            seq_len = len(decodec_characters)
            if seq_len > longest_seq_len:
                longest_seq_len = seq_len
            total_character_count += seq_len

            if seq_len <= max_seq_len:
                if sos_eos:
                    sentence_vector.insert(0, config.sos)
                    label.insert(0, config.sos)
                    sentence_vector.append(config.eos)
                    label.append(config.eos)
                sentence_vectors.append(np.asarray(sentence_vector))
                labels.append(np.asarray(label))
        f.close()
        print('Longest sentence length: %d' % longest_seq_len)
        print('Average sentence length: %.2f' % (float(total_character_count) / sentence_count))
    return sentence_vectors, labels, characters_map, reversed_map


if __name__ == '__main__':
    build_character_dictionary()
    build_sentences_set()

