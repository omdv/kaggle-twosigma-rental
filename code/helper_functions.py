#function to process one image
def process_single_image(filename):
    path = '../input/images/'+filename[0:7]+'/'+filename
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