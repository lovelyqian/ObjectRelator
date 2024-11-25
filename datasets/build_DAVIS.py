import json
import os
from PIL import Image
import numpy as np
from pycocotools.mask import encode, decode, frPyObjects
from tqdm import tqdm
import copy
from natsort import natsorted

if __name__ == '__main__':
    root_path = '/data/work2-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap'
    # splits = ['trainval', 'test-dev']
    # we only do val evaluation
    # annotation_path = os.path.join(root_path, f'2017/{splits[0]}/Annotations/480p')
    # image_path = os.path.join(root_path, f'2017/{splits[0]}/JPEGImages/480p')
    #
    # set_path = os.path.join(root_path, f'2017/{splits[0]}/ImageSets/2017/val.txt')
    save_path = os.path.join(root_path, 'egoexo_val_psalm.json')

    # val_set = []
    # with open(set_path, 'r') as f:
    #     for line in f:
    #         val_set.append(line.strip())
    #修改
    split_path = "/home/yuqian_fu/Projects/ego-exo4d-relation/correspondence/SegSwap/data/split.json"
    with open(split_path, "r") as fp:
        data_split = json.load(fp)
    # val_set = data_split["val"]
    val_set = ["51fc36b3-e769-4617-b087-3826b280cad3"]
    new_img_id = 0
    egoexo_dataset = []

    '''
    build_DAVIS.py的代码逻辑是先处理每个视频的第一帧，第一帧中的unique_instances、高宽等信息用于该视频下后续的每一帧。
    注意，unique_instances代表的是第一帧下像素的所有类别信息，如果该视频下后续的帧中有像素的类别不在unique_instances中，会报错
    '''



    for val_name in tqdm(val_set):
        #不同视角下两个相机的总路径
        vid_root_path = os.path.join(root_path, val_name)
        anno_path = os.path.join(vid_root_path, "annotation.json")
        with open(anno_path, 'r') as fp:
            annotations = json.load(fp)
        #取出本take下的所有物体
        objs = list(annotations["masks"].keys())
        print(len(objs))
        print(f"objs:{objs}")
        #将物体名称映射为id "cook":1 从1开始，区别于背景
        #TODO看看这个要不要修改为以obj_ref中的物体为准
        coco_id_to_cont_id = {coco_id: cont_id+1 for cont_id, coco_id in enumerate(objs)}

        #区分相机
        valid_cams = os.listdir(vid_root_path)
        #这一行必须加
        valid_cams.remove("annotation.json")

        #给相机排序，方便取出01开头的相机，因为序号小的相机对应的物体更多
        valid_cams = natsorted(valid_cams)
        print(valid_cams)


        ego_cams = []
        exo_cams = []
        for vc in valid_cams:
            if 'aria' in vc:
                ego_cams.append(vc)
            else:
                exo_cams.append(vc)

        ego = ego_cams[0]
        exo = exo_cams[0]
        # print(ego, exo)

        #ego、exo相机路径
        vid_ego_path = os.path.join(vid_root_path, ego)
        vid_exo_path = os.path.join(vid_root_path, exo)

        # first_frame_annotation_path = os.path.join(anno_path, sorted(os.listdir(anno_path))[0])
        # first_frame_annotation_relpath = os.path.relpath(first_frame_annotation_path, root_path)

        #setting为ego->exo，所以ego作为第一帧，即visual prompt
        #取出第一帧ego图像的id
        #获取帧的索引时，不能简单地通过os.listdir来获取，因为路径下有的图片是没有标注的，需要以注释文件的索引为准
        #路径下图片的索引和注释文件中的索引的关系：图片名称里的subsample_idx是包含annotations里的idx的 即有的图片是没有对应的注释的，所以会出现索引报错
        ego_frames = natsorted(os.listdir(vid_ego_path))
        ego_frames = [int(f.split(".")[0]) for f in ego_frames]
        print(f"vid_exo_path:{vid_exo_path}")
        exo_frames = natsorted(os.listdir(vid_exo_path))
        # exo_frames = [int(f.split(".")[0]) for f in exo_frames]
        #exo_frames是字符串形式 be like ['1' '2' '3']
        exo_frames = [f.split(".")[0] for f in exo_frames]


        #先选出两个摄像机下都出现的物体作为总的物体范围,然后再判断在该摄像机视角下每一帧中出现了哪些物体
        objs_both_have = []
        for obj in objs:
            if ego in annotations["masks"][obj].keys() and exo in annotations["masks"][obj].keys():
                objs_both_have.append(obj)

        # 获取ego注释文件中的所有索引，用于后续和exo的交叉
        # 取所有ego obj annotated_frames最长的作为基准帧数
        # 后续对exo的操作以基准帧为核心，而不是以物体为核心
        # 取ego视角下出现时间最长的物体对应的所有注释帧，作为基准帧
        obj_ref = objs_both_have[0]
        for obj in objs_both_have:
            if len(list(annotations["masks"][obj_ref][ego].keys())) < len(list(annotations["masks"][obj][ego].keys())):
                obj_ref = obj
        ego_anno_frames = natsorted(list(annotations["masks"][obj_ref][ego].keys()))
        # TODO给frames排个序
        frames = natsorted(np.intersect1d(ego_anno_frames, exo_frames))
        print(f"frames:{frames}")

        #查看每一个物体下具体有哪些帧
        # ego_anno_frames = natsorted(list(annotations["masks"][objs[0]][ego].keys()))
        # ego_anno_frames2 = natsorted(list(annotations["masks"][objs[1]][ego].keys()))
        # print(f"ego_anno_frames3:{ego_anno_frames3}")

        #TODO测试一下结果是什么样的，默认最好是字符串




        #获取ego有注释的第一帧作为参考图像
        all_ref_keys = np.asarray(
            natsorted(annotations["masks"][obj_ref][ego])
        ).astype(np.int64)
        #first_anno_key是ego有注释第一张图片的索引
        first_anno_key = str(all_ref_keys[0])
        rgb_name = f"{first_anno_key}.jpg"
        first_frame_img_path = os.path.join(vid_ego_path, rgb_name)
        first_frame_img_relpath = os.path.relpath(first_frame_img_path, root_path)

        # first_frame_annotation_img = Image.open(first_frame_annotation_path)
        # first_frame_annotation = np.array(first_frame_annotation_img)
        # height, width = first_frame_annotation.shape


       #测试查看subsample_idx和注释文件中的idx索引是否相同
        # id_sort = natsorted(os.listdir(vid_ego_path))
        # all_ref_keys = natsorted(annotations["masks"][objs[0]][ego])
        # json_output_path1 = '/home/yuqian_fu/Projects/PSALM/annotation456.json'
        # json_output_path2 = '/home/yuqian_fu/Projects/PSALM/annotation123.json'
        # with open(json_output_path1, 'w') as json_file:
        #     json.dump(id_sort, json_file)
        # with open(json_output_path2, 'w') as json_file:
        #     json.dump(all_ref_keys, json_file)
        # print(first_frame_img_id, type(first_frame_img_id))
        # print(objs[0])
        # print(ego)


        # 改为通过json文件获取大小
        height, width = annotations["masks"][obj_ref][ego][first_anno_key]["size"]
        # print(annotations["masks"][objs[0]][ego].keys())


        #np.unique存储每一帧中的所有像素类别
        # unique_instances = np.unique(first_frame_annotation)
        # unique_instances = unique_instances[unique_instances != 0]
        coco_format_annotations = []
        # for semi-supervised VOS, we use first frame's GT for input
        # for instance_value in unique_instances:
        #     binary_mask = (first_frame_annotation == instance_value).astype(np.uint8)
        #     segmentation = encode(np.asfortranarray(binary_mask))
        #     #可以直接从annotation中取出来
        #     segmentation = {
        #         'counts': segmentation['counts'].decode('ascii'),
        #         'size': segmentation['size'],
        #     }
        #     #area可能得decode搞一下
        #     area = binary_mask.sum().astype(float)
        #     coco_format_annotations.append(
        #         {
        #             'segmentation': segmentation,
        #             'area': area,
        #             'category_id': instance_value.astype(float),
        #         }
        #     )

        #统计每一帧下具体有哪些物体，这里统计的是参考帧ego的
        #追踪的物体范围以ego参考帧中的物体为准，因为你输入的mask不可能超过这个范围
        obj_list_ego = []
        for obj in objs_both_have:
            if first_anno_key in annotations["masks"][obj][ego].keys():
                obj_list_ego.append(obj)
        print(len(obj_list_ego))
        print(obj_list_ego)




        #处理ego帧中的物体mask
        for obj in obj_list_ego:
            # binary_mask = (first_frame_annotation == instance_value).astype(np.uint8)
            #TODO看看segmentation中count和size的顺序影不影响使用
            segmentation = annotations["masks"][obj][ego][first_anno_key]
            # 可以直接从annotation中取出来
            # segmentation = {
            # 'counts': segmentation['counts'].decode('ascii'),
            # 'size': segmentation['size'],
            # }
            # area可能得decode搞一下
            binary_mask = decode(segmentation)
            #TODO检查一下binary mask
            area = binary_mask.sum().astype(float)
            coco_format_annotations.append(
            {
                'segmentation': segmentation,
                'area': area,
                'category_id': float(coco_id_to_cont_id[obj]),
            }
            )

        #检查每个物体对应哪些摄像机，因为并不是每个物体对应所有的摄像机
        # for obj in objs:
        #     cams = list(annotations["masks"][obj].keys())
        #     print(f"{obj}:{cams}")


        #TODO
        for idx in frames[1:]:
            filename = f"{idx}.jpg"
            sample_img_path = os.path.join(vid_exo_path, filename)
            sample_img_relpath = os.path.relpath(sample_img_path, root_path)

            #统计每一exo帧下有哪些物体
            #有两种方式，第一种是统计该帧下在本take所有物体范围objs中出现的物体，可能会出现Found new target not in the first frame的错误
            #第二种方式是统计统计该帧下在参考帧范围obj_list_ego中出现的物体
            obj_list_exo = []
            for obj in obj_list_ego:
                if idx in annotations["masks"][obj][exo].keys():
                    obj_list_exo.append(obj)
            height, width = annotations["masks"][obj_list_exo[0]][exo][idx]["size"]
            image_info = {
                'file_name': sample_img_relpath,
                'height': height,
                'width': width,
            }

            # sample_annotation_path = os.path.join(anno_path, annfilename)
            # sample_annotation = np.array(Image.open(sample_annotation_path))
            # sample_unique_instances = np.unique(sample_annotation)
            # sample_unique_instances = sample_unique_instances[sample_unique_instances != 0]
            anns = []

            #统计本帧下物体对应的id，方便后续根据category_id调整first_frame_anns
            sample_unique_instances = [float(coco_id_to_cont_id[obj]) for obj in obj_list_exo]
            print(f"sample_unique_instances:{sample_unique_instances}")

            for obj in obj_list_exo:
                # binary_mask = (sample_annotation == instance_value).astype(np.uint8)
                assert obj in obj_list_ego, 'Found new target not in the first frame'
                segmentation = annotations["masks"][obj][exo][idx]
                binary_mask = decode(segmentation)
                area = binary_mask.sum().astype(float)
                anns.append(
                    {
                        'segmentation': segmentation,
                        'area': area,
                        'category_id': float(coco_id_to_cont_id[obj]),
                    }
                )

            #deepcopy的目的是，后续要根据本exo帧中物体的数量对参考帧的注释进行调整，防止修改原始注释
            first_frame_anns = copy.deepcopy(coco_format_annotations)
            #考虑本帧物体的数量小于参考帧的情况，仅取出参考帧中本帧有的物体的注释；但是实际情况下，有可能本帧物体的数量会大于参考帧，这时候就需要调整统计本帧下有哪些物体时，总的物体范围
            if len(anns) < len(first_frame_anns):
                first_frame_anns = [ann for ann in first_frame_anns if ann['category_id'] in sample_unique_instances]
            assert len(anns) == len(first_frame_anns)
            sample = {
                'image': sample_img_relpath,
                'image_info': image_info,
                'anns': anns,
                'first_frame_image': first_frame_img_relpath,
                'first_frame_anns': first_frame_anns,
                'new_img_id': new_img_id,
                'video_name': val_name,
            }
            egoexo_dataset.append(sample)
            new_img_id += 1

    with open(save_path, 'w') as f:
        json.dump(egoexo_dataset, f)
    print(f'Save at {save_path}. Total sample: {len(egoexo_dataset)}')
