from urllib.request import Request, urlopen
import re
import zipfile
import pickle
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, 'Data')
EXTRACT_DIR = os.path.join(THIS_DIR, 'DataE')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)

def download_top_k_played_songs(k):
    if not os.path.isdir(DATA_DIR):  # SUBJECT TO RACE CONDITION
        os.makedirs(DATA_DIR)

    downloaded_songs_full = [dI for dI in os.listdir(EXTRACT_DIR) if os.path.isdir(os.path.join(EXTRACT_DIR, dI))]
    i = 0
    total_downloaded = 0
    downloaded_songs = []
    while i < len(downloaded_songs_full):
        try:
            temp = downloaded_songs_full[i].split(')', 1)
            downloaded_songs.append(temp[1])
            if int(temp[0]) > total_downloaded:
                total_downloaded = int(temp[0])
        except:
            print('Fail @ '+str(i)+': '+downloaded_songs_full[i])
            pass
        i += 1

    titleRegEx = re.compile("<td>Song:\s(.*?)<\/td>")
    authorRegEx = re.compile("Uploaded by: <a href=\"https:\/\/beatsaver\.com\/browse\/byuser\/\d+\">(.*?)<\/a>")
    downloadsRegEx = re.compile("<li>Downloads:\s(.*?)<\/li>")
    finishedRegEx = re.compile("<li>Finished:\s(.*?)<\/li>")
    thumbsUpRegEx = re.compile("<i class=\"fas fa-thumbs-up\"><\/i>\s(.*?) \/")
    thumbsDownRegEx = re.compile("<i class=\"fas fa-thumbs-down\"><\/i>\s(.*?)"+r"\\"+"n\s*?<\/li>")
    ratingRegEx = re.compile("<li>Rating:\s(.*?)%<\/li>")

    fileNameRegEx = re.compile("has-text-right\">Version:\s([0-9]+-[0-9]+)<\/td>")
    illegalNameChars = re.compile("[\/\\:\*\?\"<>|]")
    nbDownloaded = 0
    page = 0
    while nbDownloaded < k:
        # Get the HTML page. The page has 20 files on it
        HTMLreq = Request('http://beatsaver.com/browse/played/'+str(page*20), headers={'User-Agent': 'Mozilla/5.0'})
        response = urlopen(HTMLreq)
        HTML = str(response.read())
        titleMatches = re.findall(titleRegEx, HTML)  # Extract song titles
        fileNameMatches = re.findall(fileNameRegEx, HTML)  # Extract file names (both very hacky)
        authorMatches = re.findall(authorRegEx, HTML)  # Extract author names
        downloadsMatches = re.findall(downloadsRegEx, HTML)  # Extract num downloads
        finishedMatches = re.findall(finishedRegEx, HTML)  # Extract num finished
        thumbsUpMatches = re.findall(thumbsUpRegEx, HTML)  # Extract thumb ups
        thumbsDownMatches = re.findall(thumbsDownRegEx, HTML)  # Extract thumb downs
        ratingMatches = re.findall(ratingRegEx, HTML)  # Extract rating
        for index, match in enumerate(fileNameMatches):
            # Download the corresponding ZIP file
            if nbDownloaded <= k:
                fileName = re.sub(illegalNameChars, "", titleMatches[index])
                if fileName not in downloaded_songs:
                    song_name = str(total_downloaded+1)+")"+fileName
                    dataReq = Request('http://beatsaver.com/download/'+match, headers={'User-Agent': 'Mozilla/5.0'})
                    data = urlopen(dataReq).read()
                    try:
                        pickle.dump(data, open(os.path.join(DATA_DIR, song_name+".zip"), "wb"))  # Store file
                    except:
                        song_name = str(total_downloaded+1)+')'
                        pickle.dump(data, open(os.path.join(DATA_DIR, song_name + ").zip"), "wb"))  # Safe file name choice
                    # Extract the zipfile contents to other folder
                    zip_ref = zipfile.ZipFile(os.path.join(DATA_DIR, song_name+".zip"), 'r')
                    zip_ref.extractall(os.path.join(EXTRACT_DIR, song_name))  # Need to create new folder DataE to work
                    zip_ref.close()
                    print('Extracted song ' + str(nbDownloaded) + ' of ' + str(k) + ': ' + song_name)
                    total_downloaded += 1
                    nbDownloaded += 1
                else:
                    song_name = downloaded_songs_full[downloaded_songs.index(fileName)]
                    print('Confirmed Song ' + song_name)

                # Write meta data text file
                meta_data_txt_output = os.path.join(os.path.join(EXTRACT_DIR, song_name),
                                                    'meta_data.txt')

                if not os.path.exists(meta_data_txt_output):
                    f = open(meta_data_txt_output, 'a').close() # incase doesn't exist
                if os.path.getsize(meta_data_txt_output) == 0:
                    f = open(meta_data_txt_output, 'w')
                    f.write('id: ' + fileNameMatches[index] + ';\n')
                    f.write('author: '+authorMatches[index]+';\n')
                    f.write('downloads: ' + downloadsMatches[index] + ';\n')
                    f.write('finished: ' + finishedMatches[index] + ';\n')
                    f.write('thumbsUp: ' + thumbsUpMatches[index] + ';\n')
                    f.write('thumbsDown: ' + thumbsDownMatches[index] + ';\n')
                    f.write('rating: ' + ratingMatches[index] + ';\n')
                    f.close()
                    print('Created meta_data ' + os.path.join(song_name, meta_data_txt_output))
                else:
                    print('Confirmed meta_data for ' + song_name)
        page += 1
    return fileNameMatches, titleMatches


if __name__ == "__main__":
    fileNamesA, titlesA = download_top_k_played_songs(10)
