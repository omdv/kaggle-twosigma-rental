import subprocess
import os
import json
import pandas as pd
import numpy as np
import exifread
from PIL import Image
from PIL.ExifTags import TAGS
from subprocess import check_output

class ExifTool(object):

    sentinel = "{ready}\n"

    def __init__(self, executable="/usr/local/bin/exiftool"):
        self.executable = executable

    def __enter__(self):
        self.process = subprocess.Popen(
            [self.executable, "-stay_open", "True",  "-@", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        return self

    def  __exit__(self, exc_type, exc_value, traceback):
        self.process.stdin.write("-stay_open\nFalse\n")
        self.process.stdin.flush()

    def execute(self, *args):
        args = args + ("-execute\n",)
        self.process.stdin.write(str.join("\n", args))
        self.process.stdin.flush()
        output = ""
        fd = self.process.stdout.fileno()
        while not output.endswith(self.sentinel):
            output += os.read(fd, 4096)
        return output[:-len(self.sentinel)]

    def get_metadata(self, *filenames):
        return json.loads(self.execute("-G", "-j", "-n", *filenames))


#function to process one image
def process_single_image(filename):
    path = '../input/images_sample/'+filename[0:7]+'/'+filename
    try:
        im = np.array(Image.open(path))
    except:
        res = np.empty(4)
        res[:] = np.nan
        return res
        pass

    #get dims
    if im.shape:
        width = im.shape[1]
        height = im.shape[0]
    else:
        width = np.nan
        height = np.nan

    # check if image is B&W or colored
    if len(im.shape) == 3:
        #flatten image
        im = im.transpose(2,0,1).reshape(3,-1)
    
        #brightness is simple, assign 1 if zero to avoid divide
        brg = np.amax(im,axis=0)
        brg[brg==0] = 1
        brg = np.mean(brg)
    
        #hue, same, assign 1 if zero, not working atm due to arccos
        #denom = np.sqrt((im[0]-im[1])**2-(im[0]-im[2])*(im[1]-im[2]))
        #denom[denom==0] = 1
        #hue = np.arccos(0.5*(2*im[0]-im[1]-im[2])/denom)
        
        #saturation
        sat = (brg - np.amin(im,axis=0))/brg
        sat = np.mean(sat)
    else:
        brg = np.nan
        sat = np.nan

    #return mean values
    return width,height,brg,sat

#function to process one image
def get_exif_single_image(filename):
    # path = '../input/images_sample/'+filename[0:7]+'/'+filename
    path = '/Users/om/Documents/python/Kaggle-Images/'+filename[0:7]+'/'+filename

    try:
        im = Image.open(path)
    except:
        exif = np.nan
        return exif
        pass

    try:
        exif = im._getexif()
    except:
        exif = {}
        pass
    
    #return mean values
    # f = open(path,'rb')
    # exif = exifread.process_file(f)
    return exif

def get_exif_all_images(row):
    images = row['photo_files']
    tags = [get_exif_single_image(x) for x in images]
    
    # exifread version
    # for tag in tags:
    #     for key in tag.keys():
    #         row[key] = tag[key]

    # PIL version
    for tag in tags:
        if tag:
            try:
                for (k,v) in tag.iteritems():
                    row[TAGS.get(k)]=v
            except:
                pass
    
    return row

def process_all_images(row):
    images = row['photo_files']
    res = np.empty(4)
    res[:] = np.nan
    if images:
        res = np.array([process_single_image(x) for x in images])
        res = np.nanmean(res,axis=0)
    row['img_wdt_mean'] = res[0]
    row['img_hgt_mean'] = res[1]
    row['img_brg_mean'] = res[2]
    row['img_sat_mean'] = res[3]
    return row

def description_sentiment(sentences):
    analyzer = SentimentIntensityAnalyzer()
    result = []
    for sentence in sentences:
        vs = analyzer.polarity_scores(sentence)
        result.append(vs)
    return pd.DataFrame(result).mean()

def get_all_exif(row):
    # print filenames
    with ExifTool() as e:
        try:
            metadata = e.get_metadata(*filenames)
            return metadata
        except:
            pass

if __name__ == '__main__':
    train = pd.read_json('../input/train.json')
    test = pd.read_json('../input/test.json')
    df = pd.concat([train,test])
    # df = df.sample(100)

    images = [int(x) for x in check_output(["ls", "../input/images_sample"]).\
        decode("utf8").strip().split('\n')]

    # Read the train set and choose those which have images only
    # df = df[df.listing_id.isin(images)]
    print(df.shape)

    # Add number of images
    df['n_images'] = df.apply(lambda x: len(x['photos']), axis=1)
    df['photo_files'] = df.apply(lambda x:\
        [y[29:] for y in x['photos']], axis=1)
    df['photo_files'] = df.apply(lambda x:\
        ['/Users/om/Documents/python/Kaggle-Images/'+\
        y[0:7]+'/'+y for y in x['photo_files']], axis=1)

    filenames = df.apply(lambda x: pd.Series(x['photo_files']),axis=1).\
        stack().reset_index(level=1, drop=True)
    filenames.name = 'photo_files'
    df = df.drop('photo_files', axis=1).join(filenames)

    res = get_all_exif(df['photo_files'])
    # res['listing_id'] = res.SourceFile.str.extract('([0-9]+)')

    # # Select features
    # counts = []
    # for f in res.columns:
    #     try:
    #         count = len(res[f].unique())
    #         total = len(res[f])
    #         nonnil = len(res[res[f].notnull()])
    #     except:
    #         pass
    #     counts.append({'feature':f, 'unique': count, 'nonnil' : nonnil})
    # df = pd.DataFrame(counts)

    # features_to_use = df[(df.nonnil>np.percentile(df.nonnil,75)) &\
    #     (df.unique < df.unique.max()) &\
    #     (df.unique > np.percentile(df.unique,90))].\
    #     sort_values(by='unique',ascending=False)['feature'].values.tolist()


    # features_to_use.remove('listing_id')
    # features_to_use.remove('File:Directory')

    # cor = res[features_to_use].corr()
    # cor.loc[:,:] = np.tril(cor,k=-1)
    # cor = cor.stack()
    # kk = cor[cor>0.9999]

    # for f in kk.unstack().columns:
    #     features_to_use.remove(f)