import os
import threading

from flask import Flask, request, jsonify, json
from detection.detection import detection_power
from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2
from object_detection.utils import label_map_util

from utils.config import process_config
from utils.utils import get_args
import tensorflow as tf

app = Flask(__name__)
args = get_args()
configs = process_config(args.config)


def load_model(configs):
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(configs["model"], 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph


detection_graph = load_model(configs)


def label_string(label_list):
    label_map = ''
    for label in label_list:
        label_map_string = "id:{},name:'{}'".format(label['flawId'], label['flawCode'])
        string = "item:{}".format("{" + label_map_string + "}")
        label_map += string + "\n"
    return label_map


@app.route('/detection', methods=['POST'])
def detection():
    data = request.get_data()
    json_data = json.loads(data.decode("utf-8"))
    flawCategorys = json_data["flawCategorys"]
    taskFileSet = json_data["taskFileSet"]

    label_map_string = label_string(flawCategorys)
    label_map = string_int_label_map_pb2.StringIntLabelMap()
    label_map = text_format.Merge(label_map_string, label_map)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=len(flawCategorys),
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    threading.Thread(target=detection_power, args=(taskFileSet, category_index, detection_graph, configs,)).start()
    status = {'status': 0}
    return jsonify(status)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)
