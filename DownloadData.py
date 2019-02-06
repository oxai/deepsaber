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

    titleRegEx = re.compile("<td>Song:\s(.*?)<\/td>")
    fileNameRegEx = re.compile("has-text-right\">Version:\s([0-9]+-[0-9]+)<\/td>")
    illegalNameChars = re.compile("[\/\\:\*\?\"<>|]")
    nbDownloaded = 0
    while nbDownloaded < k:
        # Get the HTML page. The page has 20 files on it
        HTMLreq = Request('http://beatsaver.com/browse/played/'+str(nbDownloaded), headers={'User-Agent': 'Mozilla/5.0'})
        response = urlopen(HTMLreq)
        HTML = str(response.read())
        titleMatches = re.findall(titleRegEx, HTML)  # Extract song titles
        fileNameMatches = re.findall(fileNameRegEx, HTML)  # Extract file names (both very hacky)
        for index, match in enumerate(fileNameMatches):
            # Download the corresponding ZIP file
            fileName = re.sub(illegalNameChars, "", titleMatches[index])
            dataReq = Request('http://beatsaver.com/download/'+match, headers={'User-Agent': 'Mozilla/5.0'})
            data = urlopen(dataReq).read()
            try:
                pickle.dump(data, open(os.path.join(DATA_DIR, str(nbDownloaded+1)+")"+fileName+".zip"), "wb"))  # Store file
            except:
                pickle.dump(data, open(os.path.join(DATA_DIR, str(nbDownloaded + 1) + ").zip"), "wb"))  # Safe file name choice
            # Extract the zipfile contents to other folder
            zip_ref = zipfile.ZipFile(os.path.join(DATA_DIR, str(nbDownloaded+1)+")"+fileName+".zip"), 'r')
            zip_ref.extractall(os.path.join(EXTRACT_DIR, str(nbDownloaded+1)+")"+fileName))  # Need to create new folder DataE to work
            zip_ref.close()
            nbDownloaded += 1
            print('Extracted song '+str(nbDownloaded)+' of '+str(k)+': '+fileName)
    return fileNameMatches, titleMatches


if __name__ == "__main__":
    fileNamesA, titlesA = download_top_k_played_songs(10)
