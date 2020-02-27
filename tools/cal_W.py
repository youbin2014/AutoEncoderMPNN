import json
import numpy as np
import os
import pickle

label_dict=['a16_race',
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
label_dir='/home/hh9665/Privacy_original/Privacy_original/train2017_annotation/'
label_list=os.listdir(label_dir)
W1=np.zeros([68,68])
#i
W2=np.zeros([68,68])
#j
W3=np.zeros([68,68])
#j|i
W4=np.zeros([68,68])
#i|j
W5=np.zeros([68,68])
#j|-i
W6=np.zeros([68,68])
#i|-j

W7=np.zeros([68,68])
#-i
W8=np.zeros([68,68])
#-j


for file in label_list:
    label_json=json.load(open(label_dir+file,'rb'))
    labels=label_json['labels']

    for i in range(len(label_dict)):
        for j in range(len(label_dict)):
            if label_dict[i] in labels:
                W1[i,j]+=1
            if label_dict[j] in labels:
                W2[i,j]+=1
            if label_dict[i] in labels and label_dict[j] in labels:
                W3[i,j]+=1
            if label_dict[i] in labels and label_dict[j] in labels:
                W4[j,i]+=1
            if label_dict[i] not in labels and label_dict[j] in labels:
                W5[i,j]+=1
            if label_dict[i] in labels and label_dict[j] not in labels:
                W6[i,j]+=1
            if label_dict[i] not in labels:
                W7[i, j] += 1
            if label_dict[j] not in labels:
                W8[i,j]+=1
Matrix1=W3/W1
Matrix2=W4/W2
Matrix3=W5/W7
Matrix4=W6/W8

pickle.dump(Matrix1,open('/home/hh9665/Desktop/CurrentProject/AutoEncoderMPNN/dataset/Matrix1','wb'))
pickle.dump(Matrix2,open('/home/hh9665/Desktop/CurrentProject/AutoEncoderMPNN/dataset/Matrix2','wb'))
pickle.dump(Matrix3,open('/home/hh9665/Desktop/CurrentProject/AutoEncoderMPNN/dataset/Matrix3','wb'))
pickle.dump(Matrix4,open('/home/hh9665/Desktop/CurrentProject/AutoEncoderMPNN/dataset/Matrix4','wb'))


