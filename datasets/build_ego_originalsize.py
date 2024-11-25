import json
import os
from PIL import Image
import numpy as np
from pycocotools.mask import encode, decode, frPyObjects
from tqdm import tqdm
import copy
from natsort import natsorted
import cv2

if __name__ == '__main__':
    root_path = '/data/work2-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap'
    #实验需改动
    save_path = os.path.join(root_path, 'egoexo_val_debug_new.json')

    #获取takes_id
    split_path = "/home/yuqian_fu/Projects/ego-exo4d-relation/correspondence/SegSwap/data/split.json"
    # with open(split_path, "r") as fp:
    #     data_split = json.load(fp)
    # val_set = data_split["val"]

    #实验需改动
    val_set = ["92b2221b-ae92-44f0-bb31-e2d27cb736d6"]
    
    #用来计数
    new_img_id = 0
    
    #用来存储json中的数据
    egoexo_dataset = []

    '''
    build_DAVIS.py的代码逻辑是先处理每个视频的第一帧，第一帧中的unique_instances、高宽等信息用于该视频下后续的每一帧。
    注意，unique_instances代表的是第一帧下像素的所有类别信息，如果该视频下后续的帧中有像素的类别不在unique_instances中，会报错
    '''

    bad_case = []

    for val_name in tqdm(val_set):
        #不同视角下两个相机的总路径
        vid_root_path = os.path.join(root_path, val_name)
        anno_path = os.path.join(vid_root_path, "annotation.json")
        with open(anno_path, 'r') as fp:
            annotations = json.load(fp)
        #取出本take下的所有物体
        # objs = list(annotations["masks"].keys())
        # 这个只在计算指标的时候有影响，且只要顺序一致就没事
        objs = natsorted(list(annotations["masks"].keys()))
        print("the total obj num are:", len(objs))
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
        print("valid_cams:", valid_cams)


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
        # vid_exo_path = os.path.join(vid_root_path, exo)


        

        #setting为ego->exo，所以ego作为第一帧，即visual prompt
        #取出第一帧ego图像的id
        #获取帧的索引时，不能简单地通过os.listdir来获取，因为路径下有的图片是没有标注的，需要以注释文件的索引为准
        #路径下图片的索引和注释文件中的索引的关系：图片名称里的subsample_idx是包含annotations里的idx的 即有的图片是没有对应的注释的，所以会出现索引报错
        ego_frames = natsorted(os.listdir(vid_ego_path))
        ego_frames = [int(f.split(".")[0]) for f in ego_frames]
        


        #先选出两个摄像机下都出现的物体作为总的物体范围,然后再判断在该摄像机视角下每一帧中出现了哪些物体
        #也可能出现objs_both_have为空的情况，这时候就需要更换exo摄像机
        objs_both_have = []
        for obj in objs:
            if ego in annotations["masks"][obj].keys() and exo in annotations["masks"][obj].keys():
                objs_both_have.append(obj)

        if len(exo_cams) > 1:
            for cam in exo_cams[1:]:
                objs_both_have_tmp = []
                for obj in objs:
                    if ego in annotations["masks"][obj].keys() and cam in annotations["masks"][obj].keys():
                        objs_both_have_tmp.append(obj)
                if len(objs_both_have_tmp) > len(objs_both_have):
                    exo = cam
                    objs_both_have = objs_both_have_tmp
        #如果没有物体范围，跳过本take
        print("objs_both_have num:", len(objs_both_have))
        if len(objs_both_have) == 0:
            bad_case.append(val_name)
            continue
        
        print(ego, exo)
        #确定exo的最终相机后，再定义exo的路径
        vid_exo_path = os.path.join(vid_root_path, exo)
        print(f"vid_exo_path:{vid_exo_path}")
        exo_frames = natsorted(os.listdir(vid_exo_path))
        # exo_frames = [int(f.split(".")[0]) for f in exo_frames]
        # exo_frames是字符串形式 be like ['1' '2' '3']
        exo_frames = [f.split(".")[0] for f in exo_frames]

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
        # first_frame_img_relpath = "ball_test/aria01_214-1/60.jpg"
        # first_frame_annotation_img = Image.open(first_frame_annotation_path)
        # first_frame_annotation = np.array(first_frame_annotation_img)
        # height, width = first_frame_annotation.shape


        # 改为通过json文件获取ego mask大小，在我们的脚本中用不上，因为ego和exo大小不一样
        # height1, width1 = annotations["masks"][obj_ref][ego][first_anno_key]["size"]
        


        #np.unique存储每一帧中的所有像素类别
        # unique_instances = np.unique(first_frame_annotation)
        # unique_instances = unique_instances[unique_instances != 0]
        #这个列表用于存储第一帧的注释信息
        coco_format_annotations = []
        

        #统计每一帧下具体有哪些物体，这里统计的是参考帧ego的
        #追踪的物体范围以ego参考帧中的物体为准，因为你输入的mask不可能超过这个范围
        #注意：有可能出现参考帧的注释中有这个物体，但是物体的mask为0的情况，这种物体得排除，因为后续对输入图片进行mask pooling的时候会出错
        obj_list_ego = []
        for obj in objs_both_have:
            if first_anno_key in annotations["masks"][obj][ego].keys():
                mask_ego = decode(annotations["masks"][obj][ego][first_anno_key])
                area_new = mask_ego.sum().astype(float)
                if area_new != 0:
                    obj_list_ego.append(obj)
        print("total obj num in ego", len(obj_list_ego))
        if len(obj_list_ego) == 0:
            bad_case.append(val_name)
            continue
        # print(obj_list_ego)



        obj_list_ego_new = []
        #处理ego帧中的物体mask
        for obj in obj_list_ego:
            #TODO看看segmentation中count和size的顺序影不影响使用
            segmentation_tmp = annotations["masks"][obj][ego][first_anno_key]
            # 可以直接从annotation中取出来
            
            # area可能得decode搞一下
            binary_mask = decode(segmentation_tmp)
            # print("original binary_mask_shape:", binary_mask.shape)

            #对解码后的mask进行缩放，使得可以匹配ego图像的大小
            h,w = binary_mask.shape
            binary_mask = cv2.resize(binary_mask, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
            #这里计算的area是resize后的mask面积
            area = binary_mask.sum().astype(float)
            if area == 0:
                # obj_list_ego.remove(obj)
                continue
            segmentation = encode(np.asfortranarray(binary_mask))
            segmentation = {
                'counts': segmentation['counts'].decode('ascii'),
                'size': segmentation["size"],
            }
            obj_list_ego_new.append(obj)
            coco_format_annotations.append(
            {
                'segmentation': segmentation,
                'area': area,
                'category_id': float(coco_id_to_cont_id[obj]),
            }
            )

        if len(obj_list_ego_new) == 0:
            bad_case.append(val_name)
            continue
        #检查每个物体对应哪些摄像机，因为并不是每个物体对应所有的摄像机
        # for obj in objs:
        #     cams = list(annotations["masks"][obj].keys())
        #     print(f"{obj}:{cams}")


        #开始处理exo相机下的每一帧
        #看看索引从1开始还是从0开始
        for idx in frames[1:]:
            filename = f"{idx}.jpg"
            sample_img_path = os.path.join(vid_exo_path, filename)
            sample_img_relpath = os.path.relpath(sample_img_path, root_path)

            #统计每一exo帧下有哪些物体
            #有两种方式，第一种是统计该帧下在本take所有物体范围objs中出现的物体，可能会出现Found new target not in the first frame的错误
            #第二种方式是统计统计该帧下在参考帧范围obj_list_ego中出现的物体
            obj_list_exo = []
            for obj in obj_list_ego_new:
                if idx in annotations["masks"][obj][exo].keys():
                    mask_exo = decode(annotations["masks"][obj][exo][idx])
                    area_exo = mask_exo.sum().astype(float)
                    if area_exo != 0:
                        obj_list_exo.append(obj)
                        
            #检查exo下每一帧的物体数量，也会碰到有的帧一个物体也没有，这种直接跳过
            print("total obj num in exo", len(obj_list_exo))
            if len(obj_list_exo) == 0:
                continue

            height, width = annotations["masks"][obj_list_exo[0]][exo][idx]["size"]
            # print("original exo mask_shape:" ,height,width)
            image_info = {
                'file_name': sample_img_relpath,
                'height': height//4,
                'width': width//4,
            }

            
            anns = []

            obj_list_exo_new = []
            for obj in obj_list_exo:
                assert obj in obj_list_ego_new, 'Found new target not in the first frame'
                segmentation_tmp = annotations["masks"][obj][exo][idx]
                binary_mask = decode(segmentation_tmp)
                # print("original ego binary_mask_shape", binary_mask.shape)
                h, w = binary_mask.shape
                binary_mask = cv2.resize(binary_mask, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
                # print("binary_mask", binary_mask.shape)
                area = binary_mask.sum().astype(float)
                if area == 0:
                    continue
                segmentation = encode(np.asfortranarray(binary_mask))
                segmentation = {
                    'counts': segmentation['counts'].decode('ascii'),
                    'size': segmentation['size'],
                }
                obj_list_exo_new.append(obj)
                anns.append(
                    {
                        'segmentation': segmentation,
                        'area': area,
                        'category_id': float(coco_id_to_cont_id[obj]),
                    }
                )

            if len(obj_list_exo_new) == 0:
                continue

            # 统计本帧下物体对应的id，方便后续根据category_id调整first_frame_anns
            sample_unique_instances = [float(coco_id_to_cont_id[obj]) for obj in obj_list_exo_new]
            # 查看每一帧下有哪些物体
            print(f"sample_unique_instances in {idx}:{sample_unique_instances}")

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


    print(bad_case)
    
    with open(save_path, 'w') as f:
        json.dump(egoexo_dataset, f)
    print(f'Save at {save_path}. Total sample: {len(egoexo_dataset)}')
