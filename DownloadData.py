from urllib.request import Request, urlopen
import re
import zipfile
import pickle
def downloadTopKPlayedSongs(k):
    titleRegEx = re.compile("<td>Song:\s(.*?)<\/td>")
    fileNameRegEx = re.compile("has-text-right\">Version:\s([0-9]+-[0-9]+)<\/td>")
    illegalNameChars = re.compile("[\/\\:\*\?\"<>|]")
    nbDownloaded = 0
    while nbDownloaded < k:
        # Get the HTML page. The page has 20 files on it
        HTMLreq = Request('http://beatsaver.com/browse/played/'+str(nbDownloaded), headers={'User-Agent': 'Mozilla/5.0'})
        response = urlopen(HTMLreq)
        HTML = str(response.read())
        titleMatches = re.findall(titleRegEx,HTML)  # Extract song titles
        fileNameMatches = re.findall(fileNameRegEx,HTML)  # Extract file names (both very hacky)
        for index, match in enumerate(fileNameMatches):
            # Download the corresponding ZIP file
            fileName = re.sub(illegalNameChars, "", titleMatches[index])
            dataReq = Request('http://beatsaver.com/download/'+match, headers={'User-Agent': 'Mozilla/5.0'})
            data = urlopen(dataReq).read()
            try:
                pickle.dump(data, open("Data/"+str(nbDownloaded+1)+")"+fileName+".zip", "wb"))  # Store file
            except:
                pickle.dump(data, open("Data/" + str(nbDownloaded + 1) + ").zip", "wb"))  # Safe file name choice
            # Extract the zipfile contents to other folder
            zip_ref = zipfile.ZipFile("Data/"+str(nbDownloaded+1)+")"+fileName+".zip", 'r')
            zip_ref.extractall("DataE/"+str(nbDownloaded+1)+")"+fileName)  # Need to create new folder DataE to work
            zip_ref.close()
            nbDownloaded += 1
            print(nbDownloaded)
            if nbDownloaded == k: # If Downloaded target number of zip files, stop
                return fileNameMatches, titleMatches


fileNamesA, titlesA = downloadTopKPlayedSongs(1000)