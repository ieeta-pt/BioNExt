def decoder(samples, offsets, text=None, padding=64):
    # assert len(model_outputs) == len(offsets)
    prev = 0
    temp = []
    data = []


    document_outputs = []
    document_offsets = []
    # construct the document
    for i in range(len(samples) - 1):
        offset_padding = padding - (512 - len(offsets[i]))
        
        document_outputs.extend(samples[i][padding:-padding])
        document_offsets.extend(offsets[i][padding:-offset_padding])
    # account for the last arrary
    document_outputs.extend(samples[-1][padding:-1])
    document_offsets.extend(offsets[-1][padding:-1])
    curr_label = 0
    labels = []
    # print(len(document_outputs), len(document_offsets))
    for label, offset in zip(document_outputs, document_offsets):
        # anything to B = 1
        if label != 0:
            curr_label = (label-1)//2
            label = label%2
            if label == 0:
                label = 2
        if (label == 1) or (prev == 0 and label == 2):
            if len(temp) != 0:
                data.append(temp)
                labels.append(curr_label)
            temp = [offset]
        # B or I to Is
        # since OII is now valid
        elif (label == 2):
            temp.append(offset)
        # B or I to O
        elif prev != 0 and label == 0:
            if len(temp) != 0:
                data.append(temp)
                labels.append(curr_label)
            temp = []
        prev = label

    # incase something has not been flushed form the buffer
    if len(temp) != 0:
        data.append(temp)
        labels.append(curr_label)

    text_ranges = []
    spans = []
    if text is not None:
        for i,j in zip(data,labels):
            text_ranges.append(text[i[0][0]:i[-1][1]])
            spans.append((i[0][0], i[-1][1], int(j)))
    else:
        for i,j in zip(data,labels):
            spans.append((i[0][0], i[-1][1], int(j)))

    return {"span": spans, "text": text_ranges, "labels": labels}
