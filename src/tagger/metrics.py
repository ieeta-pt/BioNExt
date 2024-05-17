from decoder import decoder
from trainer import NERevalPrediction
from collections import defaultdict
from sklearn.metrics import confusion_matrix, f1_score
# from seqeval.metrics import f1_score
from itertools import chain
import numpy as np
label2id = {"GeneOrGeneProduct": 0, "DiseaseOrPhenotypicFeature": 1, "ChemicalEntity": 2, "SequenceVariant": 3,
            "OrganismTaxon": 4, "CellLine": 5, "Disease":1}


def f1PR(tp, fn, fp):
    precision = 0 if tp == 0 else tp / (tp + fp)
    recall = 0 if tp == 0 else tp / (tp + fn)
    f1 = 0 if precision == 0 and recall == 0 else (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


class NERMetrics:

    def __init__(self, tagger, context_size=64):
        self.context_size = context_size
        self.tagger = tagger

    def __call__(self, evaluationOutput: NERevalPrediction):
        # evaluationOutput.metadata = {"doc_id", "sequence_id", "offsets", "og_annotations"}
        # evaluationOutput.predictions
        doc_gs = {}
        documents = {}
        padding = self.context_size
        doc_gs_labels = {}
        # reconsturct the document in the correct order
        for i in range(len(evaluationOutput.metadata)):
            doc_id = evaluationOutput.metadata[i]['doc_id']
            if doc_id not in documents.keys():
                documents[doc_id] = {}
                doc_gs[doc_id] = [(ann['start_span'], ann['end_span'], label2id[ann['label']]) for ann in
                                  evaluationOutput.metadata[i]["og_annotations"]]
                doc_gs_labels[doc_id] = [label2id[ann['label']] for ann in evaluationOutput.metadata[i]["og_annotations"]]
                # run 1 time is enough for this stuff

            documents[doc_id][evaluationOutput.metadata[i]['sequence_id']] = {
                'labels': evaluationOutput.predictions[i].tolist(),
                'offsets': evaluationOutput.metadata[i]['offsets']}

        print("DOCUMENTS:", len(documents))

      
        true_data = []#defaultdict(list)

        predicted_data = []#defaultdict(list)
        # decode each set of labels and store the offsets
        for doc in documents.keys():
            current_doc = [documents[doc][seq]['labels'] for seq in sorted(documents[doc].keys())]
            current_offsets = [documents[doc][seq]['offsets'] for seq in sorted(documents[doc].keys())]
            decoded_data = decoder(current_doc, current_offsets, padding=padding)

            predictions_per_label = defaultdict(list)
            true_data_per_label = defaultdict(list)

            for span in decoded_data["span"]:
                predictions_per_label[span[2]].append(span)

            for span in doc_gs[doc]:
                true_data_per_label[span[2]].append(span)
                
            # predicted_data.append(tuple(decoded_data["span"].extend(decoded_data["labels"])))
            # true_data.append(tuple(doc_gs[doc].extend(doc_gs_labels[doc])))
            # predicted_labels.append(decoded_data["labels"])

            predicted_data.append(predictions_per_label)
            true_data.append(true_data_per_label)

        assert len(predicted_data) == len(true_data)

        doc_macro_f1 = []
        doc_macro_p = []
        doc_macro_r = []

        tp = fn = fp = 0
        
        for doc_index in range(len(predicted_data)):
            macro_c_f1, macro_c_p, macro_c_r = 0, 0, 0
            for label_id in range(6):
                
                predicted_entities = predicted_data[doc_index][label_id] # (start, end, label (int))
                true_entities = true_data[doc_index][label_id]

                _tp = len(set(true_entities).intersection(set(predicted_entities)))
                _fn = len(set(true_entities).difference(set(predicted_entities)))
                _fp = len(set(predicted_entities).difference(set(true_entities)))

                tp += _tp
                fn += _fn
                fp += _fp
                
                class_f1, class_p, class_r = f1PR(_tp, _fn, _fp)
                macro_c_f1 += class_f1
                macro_c_p += class_p
                macro_c_r += class_r

            macro_c_f1 /= 6
            macro_c_p /= 6
            macro_c_r /= 6

            doc_macro_f1.append(macro_c_f1)
            doc_macro_p.append(macro_c_p)
            doc_macro_r.append(macro_c_r)
    
        macro_f1 = sum(doc_macro_f1)/len(doc_macro_f1)
        macro_p = sum(doc_macro_p)/len(doc_macro_f1)
        macro_r = sum(doc_macro_r)/len(doc_macro_f1)

        micro_f1, micro_p, micro_r = f1PR(tp, fn, fp)

        print("NER Micro F1:", micro_f1)

        #fix the return

        ## label_wise_metrics
        true_labels = []
        for i,md in enumerate(evaluationOutput.metadata):
            labels = self.tagger(md)["labels"]
            true_labels.append(labels + [0]*(512-len(labels)))
            evaluationOutput.predictions[i] = list(map(round, evaluationOutput.predictions[i].tolist())) + [0]*(512-len(evaluationOutput.predictions[i]))
        
        cm = confusion_matrix(list(chain.from_iterable(true_labels)), list(chain.from_iterable(evaluationOutput.predictions)))

        cm[0,0] = 0
        diag = np.diagonal(cm)
        tp = diag.sum()

        fn_tp = cm.sum(0)
        fp_tp = cm.sum(1)

        macro_label_recall = (diag / fn_tp)[1:]
        macro_label_recall[np.isnan(macro_label_recall)] = 0
        macro_label_precision = (diag / fp_tp)[1:]
        macro_label_precision[np.isnan(macro_label_precision)] = 0
        
        macro_label_f1 = (2 * macro_label_precision * macro_label_recall) / (macro_label_precision + macro_label_recall)
        macro_label_f1[np.isnan(macro_label_f1)] = 0
        
        return {"microF1": micro_f1,
                "microP": micro_p,
                "microR": micro_r,
                "macroF1": macro_f1,
                "macroP": macro_p,
                "macroR": macro_r,
                "macroP_label": float(macro_label_precision.mean()),
                "macroR_label": float(macro_label_recall.mean()),
                "macroF1_label": float(macro_label_f1.mean()),
                }



class NERMetricsNew:

    def __init__(self, tagger, context_size=64):
        self.context_size = context_size
        self.tagger = tagger

    def __call__(self, evaluationOutput: NERevalPrediction):
        # evaluationOutput.metadata = {"doc_id", "sequence_id", "offsets", "og_annotations"}
        # evaluationOutput.predictions
        doc_gs = {}
        documents = {}
        padding = self.context_size
        doc_gs_labels = {}
        # reconsturct the document in the correct order
        for i in range(len(evaluationOutput.metadata)):
            doc_id = evaluationOutput.metadata[i]['doc_id']
            if doc_id not in documents.keys():
                documents[doc_id] = {}
                doc_gs[doc_id] = [(ann['start_span'], ann['end_span'], label2id[ann['label']]) for ann in
                                  evaluationOutput.metadata[i]["og_annotations"]]
                doc_gs_labels[doc_id] = [label2id[ann['label']] for ann in evaluationOutput.metadata[i]["og_annotations"]]
                # run 1 time is enough for this stuff

            documents[doc_id][evaluationOutput.metadata[i]['sequence_id']] = {
                'labels': evaluationOutput.predictions[i].tolist(),
                'offsets': evaluationOutput.metadata[i]['offsets']}

        print("DOCUMENTS:", len(documents))

      
        true_data = []#defaultdict(list)

        predicted_data = []#defaultdict(list)
        # decode each set of labels and store the offsets
        for doc in documents.keys():
            current_doc = [documents[doc][seq]['labels'] for seq in sorted(documents[doc].keys())]
            current_offsets = [documents[doc][seq]['offsets'] for seq in sorted(documents[doc].keys())]
            decoded_data = decoder(current_doc, current_offsets, padding=padding)

            predictions_per_label = defaultdict(list)
            true_data_per_label = defaultdict(list)

            for span in decoded_data["span"]:
                predictions_per_label[span[2]].append(span)

            for span in doc_gs[doc]:
                true_data_per_label[span[2]].append(span)
                
            # predicted_data.append(tuple(decoded_data["span"].extend(decoded_data["labels"])))
            # true_data.append(tuple(doc_gs[doc].extend(doc_gs_labels[doc])))
            # predicted_labels.append(decoded_data["labels"])

            predicted_data.append(predictions_per_label)
            true_data.append(true_data_per_label)

        assert len(predicted_data) == len(true_data)

        doc_macro_f1 = []
        doc_macro_p = []
        doc_macro_r = []

        tp = fn = fp = 0
        
        for doc_index in range(len(predicted_data)):
            macro_c_f1, macro_c_p, macro_c_r = 0, 0, 0
            for label_id in range(6):
                
                predicted_entities = predicted_data[doc_index][label_id] # (start, end, label (int))
                true_entities = true_data[doc_index][label_id]

                _tp = len(set(true_entities).intersection(set(predicted_entities)))
                _fn = len(set(true_entities).difference(set(predicted_entities)))
                _fp = len(set(predicted_entities).difference(set(true_entities)))

                tp += _tp
                fn += _fn
                fp += _fp
                
                class_f1, class_p, class_r = f1PR(_tp, _fn, _fp)
                macro_c_f1 += class_f1
                macro_c_p += class_p
                macro_c_r += class_r

            macro_c_f1 /= 6
            macro_c_p /= 6
            macro_c_r /= 6

            doc_macro_f1.append(macro_c_f1)
            doc_macro_p.append(macro_c_p)
            doc_macro_r.append(macro_c_r)
    
        macro_f1 = sum(doc_macro_f1)/len(doc_macro_f1)
        macro_p = sum(doc_macro_p)/len(doc_macro_f1)
        macro_r = sum(doc_macro_r)/len(doc_macro_f1)

        micro_f1, micro_p, micro_r = f1PR(tp, fn, fp)

        print("NER Micro F1:", micro_f1)

        #fix the return

        ## label_wise_metrics
        true_labels = []
        for i,md in enumerate(evaluationOutput.metadata):
            labels = self.tagger(md)["labels"]
            true_labels.append(labels + [0]*(512-len(labels)))
            evaluationOutput.predictions[i] = list(map(round, evaluationOutput.predictions[i].tolist())) + [0]*(512-len(evaluationOutput.predictions[i]))
        
        cm = confusion_matrix(list(chain.from_iterable(true_labels)), list(chain.from_iterable(evaluationOutput.predictions)))

        cm[0,0] = 0
        diag = np.diagonal(cm)
        tp = diag.sum()

        fn_tp = cm.sum(0)
        fp_tp = cm.sum(1)

        macro_label_recall = (diag / fn_tp)[1:]
        macro_label_recall[np.isnan(macro_label_recall)] = 0
        macro_label_precision = (diag / fp_tp)[1:]
        macro_label_precision[np.isnan(macro_label_precision)] = 0
        
        macro_label_f1 = (2 * macro_label_precision * macro_label_recall) / (macro_label_precision + macro_label_recall)
        macro_label_f1[np.isnan(macro_label_f1)] = 0
        
        return {"microF1": micro_f1,
                "microP": micro_p,
                "microR": micro_r,
                "macroF1": macro_f1,
                "macroP": macro_p,
                "macroR": macro_r,
                "macroP_label": float(macro_label_precision.mean()),
                "macroR_label": float(macro_label_recall.mean()),
                "macroF1_label": float(macro_label_f1.mean()),
                }
