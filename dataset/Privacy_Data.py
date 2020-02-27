import os
import random
from PIL import Image
import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms
import json
import torch

class Privacy_Data(data.Dataset):
    def __init__(self, transform=None, train=True, test=False, valid=False):
        self.test = test
        self.train = train
        self.valid = valid
        self.transform = transform
        self.dir_name='/home/hh9665/Privacy_original/Privacy_original/'
        self.label_dict=['a16_race',
'a17_color',
'a19_name_full',
'a1_age_approx',
'a46_occupation',
'a4_gender',
'a5_eye_color',
'a66_rel_professional',
'a6_hair_color',
'a9_face_complete',
'a0_safe',
'a12_semi_nudity',
'a38_ticket',
'a57_culture',
'a68_rel_spectators',
'a48_occassion_work',
'a65_rel_social',
'a27_marital_status',
'a2_weight_approx',
'a3_height_approx',
'a60_occassion_personal',
'a10_face_partial',
'a11_tattoo',
'a56_sexual_orientation',
'a61_opinion_general',
'a69_rel_views',
'a75_address_current_partial',
'a73_landmark',
'a59_sports',
'a64_rel_personal',
'a70_education_history',
'a25_nationality',
'a31_passport',
'a13_full_nudity',
'a74_address_current_complete',
'a82_date_time',
'a39_disability_physical',
'a49_phone',
'a24_birth_date',
'a32_drivers_license',
'a78_address_home_complete',
'a8_signature',
'a58_hobbies',
'a18_ethnic_clothing',
'a43_medicine',
'a20_name_first',
'a92_email_content',
'a67_rel_competitors',
'a21_name_last',
'a99_legal_involvement',
'a90_email',
'a41_injury',
'a7_fingerprint',
'a55_religion',
'a104_license_plate_partial',
'a79_address_home_partial',
'a37_receipt',
'a62_opinion_political',
'a97_online_conversation',
'a103_license_plate_complete',
'a26_handwriting',
'a102_vehicle_ownership',
'a33_student_id',
'a35_mail',
'a30_credit_card',
'a85_username',
'a23_birth_city',
'a29_ausweis']
        # imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # test1: data/test1/8973.jpg
        # train: data/train/cat.10004.jpg

        # imgs_num = len(imgs)

        if self.test:
            dir=self.dir_name+'test2017_rgb'
            self.imgs = [os.path.join(dir,img) for img in os.listdir(dir)]
        if self.train:
            dir = self.dir_name + 'train2017_rgb'
            # dir = self.dir_name + 'samples'
            self.imgs = [os.path.join(dir, img) for img in os.listdir(dir)]
        if self.valid:
            dir = self.dir_name + 'val2017_rgb'
            self.imgs = [os.path.join(dir, img) for img in os.listdir(dir)]

    def __getitem__(self, index):
        img_path = self.imgs[index]
        file_name=img_path.split('j')[0].split('/')[-1]
        if self.test:
            dir=self.dir_name+'test2017_annotation/'
            with open(dir + file_name + 'json', 'r') as label_file:
                label_json = json.load(label_file)
                multi_label = np.zeros(68)
                for lab in label_json['labels']:
                    for i in range(len(self.label_dict)):
                        if lab == self.label_dict[i]:
                            multi_label[i]=1
        if self.train:
            dir = self.dir_name + 'train2017_annotation/'
            with open(dir + file_name + 'json', 'r') as label_file:
                label_json = json.load(label_file)
                multi_label = np.zeros(68)
                for lab in label_json['labels']:
                    for i in range(len(self.label_dict)):
                        if lab == self.label_dict[i]:
                            multi_label[i]=1
        if self.valid:
            dir = self.dir_name + 'val2017_annotation/'
            with open(dir + file_name + 'json', 'r') as label_file:
                label_json = json.load(label_file)
                multi_label = np.zeros(68)
                for lab in label_json['labels']:
                    for i in range(len(self.label_dict)):
                        if lab == self.label_dict[i]:
                            multi_label[i]=1
        data = Image.open(img_path)
        # data = data.convert('RGB')
        data = self.transform(data)
        # label=torch.from_numpy(multi_label)
        return data, multi_label
    #
    # def random_process(self,image):
    #


    def __len__(self):
        return len(self.imgs)
