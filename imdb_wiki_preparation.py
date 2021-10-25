import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import random
import os
import csv
import cv2
import argparse as argparse
from scipy.io import loadmat
from datetime import datetime
from tqdm import tqdm


headers = ['filename', 'age', 'gender']

'''Calculates age by getting difference between year taken & year of birth'''
'''Need to deduct 1 year"s worth of days before converting to datetime since ordinal format is "location-based"'''
'''If birth year is in later half of the year, we don"t consider prior days in the same year as part of the age, hence we deduct 1 year from age'''
def calc_age(taken,dob):
    birth = datetime.fromordinal(max(int(dob)-366,1))
    if birth.month < 7:
        return taken-birth.year
    else:
        return taken-birth.year-1

'''Load matlab file from path that contains all the labels'''
def load_db(mat_path):
    db = loadmat(mat_path)['imdb'][0,0]
    no_of_records = len(db['face_score'][0])
    return db, no_of_records

'''Read matlab file & extract all meta data'''
'''Age is then calculated using the dob & taken years'''
def get_meta(db):
    full_path = db['full_path'][0]
    dob = db['dob'][0]
    gender = db['gender'][0]
    photo_taken = db['photo_taken'][0]
    face_score = db['face_score'][0]
    second_face_score = db['second_face_score'][0]
    age = [calc_age(photo_taken[i],dob[i]) for i in range(len(dob))]
    return full_path, dob, gender, photo_taken, face_score, second_face_score, age

'''Creates output path & crop folder'''
'''Loads matlab file & gets number of records'''
'''Shuffle indices & do train_test split'''
'''Create train & val writers'''
def main(input_db,photo_dir,output_dir,min_score,img_size,split_ratio):
    crop_dir = os.path.join(output_dir,'crop')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)
    db, no_of_records = load_db(input_db)
    indices = list(range(no_of_records))
    random.shuffle(indices)
    train_indices = indices[:int(len(indices)*split_ratio)]
    test_indices = indices[int(len(indices)*split_ratio):]

    train_csv = open(os.path.join(output_dir,'train.csv'),'w')
    train_writer = csv.writer(train_csv,delimiter=',')
    train_writer.writerow(headers)
    val_csv = open(os.path.join(output_dir,'val.csv'),'w')
    val_writer = csv.writer(val_csv,delimiter=',')
    val_writer.writerow(headers)

    clean_and_resize(db, photo_dir, train_indices, min_score, img_size, train_writer, crop_dir)

    clean_and_resize(db, photo_dir, test_indices, min_score, img_size, val_writer, crop_dir)

'''For each image, create path in crop directory & its file path in photo directory'''
'''Check conditions for each image'''
'''If image fulfills all conditions, write filename, age & gender to csv'''
def clean_and_resize(db,photo_dir,indices,min_score,img_size,writer,crop_dir):
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(db)
    for i in tqdm(indices):
        filename = str(full_path[i][0])
        if not os.path.exists(os.path.join(crop_dir,os.path.dirname(filename))):
            os.makedirs(os.path.join(crop_dir,os.path.dirname(filename)))
        img_path = os.path.join(photo_dir,filename)

        if face_score[i] < min_score:
            continue
        if np.isnan(second_face_score[i]) & (second_face_score[i] > 0):
            continue
        if ~(0 <= age[i] <= 100):
            continue
        if np.isnan(gender[i]):
            continue

        img_gender = int(gender[i])
        img_age = int(age[i])
        img = cv2.imread(img_path)
        crop = cv2.resize(img, (img_size,img_size))
        crop_filepath = os.path.join(crop_dir,filename)
        cv2.imwrite(crop_filepath,crop)
        writer.writerow([filename,img_age,img_gender])


parser = argparse.ArgumentParser()
parser.add_argument('--db_path',required=False,default='imdb_crop/imdb_crop/imdb',help='path to matlab file')
parser.add_argument('--photo_dir',required=False,default='imdb_crop/imdb_crop',help='path to image directory')
parser.add_argument('--output_dir',required=True,help='path to output directory')
parser.add_argument('--min_score',required=False,type=float, default=1.0,help='min threshold for images')
parser.add_argument('--img_size',required=False,type=int,default=200)
parser.add_argument('--split_ratio',required=False,type=float,default=0.7)
args = parser.parse_args()

main(input_db=args.db_path, photo_dir=args.photo_dir, output_dir=args.output_dir,
         min_score=args.min_score, img_size=args.img_size, split_ratio=args.split_ratio)
