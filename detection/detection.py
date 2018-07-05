# import matplotlib
# matplotlib.use('Agg')
import cv2
import json

import numpy as np
import os
import requests
import tensorflow as tf

from detection import visualization_utils as vis_util


def run_inference_for_single_image(image, sess):
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in
                        op.outputs}
    tensor_dict = {}
    for key in ['detection_boxes', 'detection_scores', 'detection_classes']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[
                key] = tf.get_default_graph().get_tensor_by_name(
                tensor_name)

    image_tensor = tf.get_default_graph().get_tensor_by_name(
        'image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                           feed_dict={
                               image_tensor: np.expand_dims(image, 0)})

    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][
        0]
    return output_dict


def detection_power(taskFileSet, category_index, detection_graph, configs):
    with detection_graph.as_default():
        with tf.Session() as sess:
            result_list = []
            for i in range(len(taskFileSet)):
                j = i + 1
                taskFile = taskFileSet[i]
                path = taskFile["filePath"]
                print(path)
                file_name = os.path.basename(path)
                save_path = os.path.join(taskFile["flawFilePath"], file_name)
                image_np = cv2.imread(path)
                output_dict = run_inference_for_single_image(image_np, sess)

                _, probabilities = vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=1)
                cv2.imwrite(save_path, image_np)
                taskFile.pop("filePath")
                for probability in probabilities:
                    taskFile["flawFilePath"] = configs["local_images"] + save_path
                    taskFile["patrolTowerPart"] = "开口销"
                    taskFile["flawId"] = int(probability["class"])
                    result_list.append(taskFile)
                    print(taskFile)
                try:
                    requests.get(configs["local"] + "/api/updatePatrolTaskPercent/{}/{}".format(taskFile["taskId"],
                                                                                                j * 100 // len(
                                                                                                    taskFileSet)))
                except Exception:
                    print("updatePatrolTaskPercent")
                    continue
            try:
                headers = {'content-type': 'application/json'}
                r = requests.post(configs["local"] + "/api/saveFlawInfo", data=json.dumps(result_list), headers=headers)
                print("code:", r.status_code)
            except Exception as e:
                print("saveFlawInfo:", str(e))
            print("finish!")

#
# if __name__ == '__main__':
#     image_np, result_list = detection_micro(
#         "E:/北控水务/图像标注软件培训/镜检图片_标注/镜检图"
#         "片/镜检图片_镜检照片_东部大区专家系统资料_2建工环"
#         "境_北区厂_20180305草履虫一期.jpg")
#     print(result_list)
