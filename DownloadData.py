from urllib.request import Request, urlopen
import re
import zipfile
import pickle
import os
import html

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, 'Data')
EXTRACT_DIR = os.path.join(THIS_DIR, 'DataE')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)

def download_top_k_played_songs(k):
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
    thumbsDownRegEx = re.compile("<i class=\"fas fa-thumbs-down\"><\/i>\s(.*?)\s*?<\/li>")
    ratingRegEx = re.compile("<li>Rating:\s(.*?)%<\/li>")
    fileNameRegEx = re.compile("has-text-right\">Version:\s([0-9]+-[0-9]+)<\/td>")
    illegalNameChars = re.compile("[\/\\:\*\?\"<>|]")
    nbDownloaded = 0
    page = 0
    while nbDownloaded < k:
        # Get the HTML page. The page has 20 files on it
        HTMLreq = Request('http://beatsaver.com/browse/played/'+str(page*20), headers={'User-Agent': 'Mozilla/5.0'})
        response = urlopen(HTMLreq)
        HTML = str(response.read().decode())
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
                fileName = re.sub(illegalNameChars, "", html.unescape(titleMatches[index]))
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
                    print('Created meta_data ' + os.path.join(song_name, meta_data_txt_output))
                f = open(meta_data_txt_output, 'w')
                f.write('id: ' + html.unescape(fileNameMatches[index]) + '\n')
                f.write('title: ' + html.unescape(titleMatches[index]) + '\n')
                f.write('author: ' + html.unescape(authorMatches[index]) + '\n')
                f.write('downloads: ' + html.unescape(downloadsMatches[index]) + '\n')
                f.write('finished: ' + html.unescape(finishedMatches[index]) + '\n')
                f.write('thumbsUp: ' + html.unescape(thumbsUpMatches[index]) + '\n')
                f.write('thumbsDown: ' + html.unescape(thumbsDownMatches[index]) + '\n')
                f.write('rating: ' + html.unescape(ratingMatches[index]) + '\n')
                f.close()
                print('Updated meta_data for ' + song_name)
        page += 1
    return fileNameMatches, titleMatches

def update_meta_data_for_downloaded_songs():
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
            print('Fail @ ' + str(i) + ': ' + downloaded_songs_full[i])
            pass
        i += 1
    titleRegEx = re.compile("<td>Song:\s(.*?)<\/td>")
    authorRegEx = re.compile("Uploaded by: <a href=\"https:\/\/beatsaver\.com\/browse\/byuser\/\d+\">(.*?)<\/a>")
    downloadsRegEx = re.compile("<li>Downloads:\s(.*?)<\/li>")
    finishedRegEx = re.compile("<li>Finished:\s(.*?)<\/li>")
    thumbsUpRegEx = re.compile("<i class=\"fas fa-thumbs-up\"><\/i>\s(.*?) \/")
    thumbsDownRegEx = re.compile("<i class=\"fas fa-thumbs-down\"><\/i>\s(.*?)" + r"\\" + "n\s*?<\/li>")
    ratingRegEx = re.compile("<li>Rating:\s(.*?)%<\/li>")
    fileNameRegEx = re.compile("has-text-right\">Version:\s([0-9]+-[0-9]+)<\/td>")
    illegalNameChars = re.compile("[\/\\:\*\?\"<>|]")

    for i in range(total_downloaded):
        meta_data_txt_output = os.path.join(os.path.join(EXTRACT_DIR, downloaded_songs_full[i]),
                                            'meta_data.txt')

        if os.path.exists(meta_data_txt_output) and os.path.getsize(meta_data_txt_output) > 0:
            num_lines = sum(1 for line in open(meta_data_txt_output))
            f = open(meta_data_txt_output, 'r')
            id = f.readline().split(': ')[1].split(';')[0]
            title = f.readline().split(': ')[1].split(';')[0]
            author = f.readline().split(': ')[1].split(';')[0]
            downloads = f.readline().split(': ')[1].split(';')[0]
            finished = f.readline().split(': ')[1].split(';')[0]
            thumbsUp = f.readline().split(': ')[1].split(';')[0]
            thumbsDown = f.readline().split(': ')[1].split(';')[0]
            try:
                rating = f.readline().split(': ')[1].split(';')[0]
            except:
                print('fail')
        else:
            id = None
            author = None
            search_key = None

        match_found = False
        if id is not None:
            req_address = 'http://beatsaver.com/browse/detail/'+id
            HTMLreq = Request(req_address, headers={'User-Agent': 'Mozilla/5.0'})
            response = urlopen(HTMLreq)
            HTML = str(response.read())
            thumbsUpRegEx = re.compile("<span>Up\s(.*?)<\/span>")
            thumbsDownRegEx = re.compile("<span>Down\s(.*?)<\/span>")
            match_found = True
            match_id = 0
            # searching = True
            # title_searched = False
            # author_searched = False
            # page = 0
            # word_id = 0
            # if search_key is None:
            #     search_key = downloaded_songs[i]
            # # Find correct beatsaver file
            # while searching:
            #     req_address = 'http://beatsaver.com/search/all/' + str(page * 40) + '?key=' + search_key
            #     HTMLreq = Request(req_address, headers={'User-Agent': 'Mozilla/5.0'})
            #     response = urlopen(HTMLreq)
            #     HTML = str(response.read())
            #     fileNameMatches = re.findall(fileNameRegEx, HTML)  # Extract file names (both very hacky)
            #     if id not in fileNameMatches:
            #         next_page = re.search("Next Page")
            #         if next_page:
            #             page += 1  # go to next page
            #         else:
            #             if len(title) >= 3 and title_searched is False:
            #                 print('File not found via ' + search_key[i])
            #                 page = 0
            #                 search_key = title
            #                 title_searched = True
            #             if len(author) >= 3 and author_searched is False:  # backup; song name not always found
            #                 print('File not found via ' + search_key[i])
            #                 page = 0
            #                 search_key = author
            #                 author_searched = True
            #             elif word_id < len(downloaded_songs[i].split(' '))-1:  # desperate last measure, searching individual words
            #                 print('File not found via ' + search_key)
            #                 page = 0
            #                 search_key = downloaded_songs[i].split(' ')[word_id]
            #                 while len(search_key) < 3:
            #                     word_id += 1
            #                     if word_id >= len(downloaded_songs[i].split(' '))-1:
            #                         searching = False
            #                         break
            #                     search_key = downloaded_songs[i].split(' ')[word_id]
            #                 word_id += 1
            #             else:
            #                 print('File not found via ' + search_key)
            #                 print('Search methods exhausted, giving up.')
            #                 searching = False
            #     else:
            #         searching = False
            #         match_found = True
            #         match_id = fileNameMatches.index(id)
        else:
            print('No existing meta_file.')
            print('Current approach to overcome this is unstable.')
            # print('No existing meta_file, searching by folder name and taking first candidate.')
            # search_key = downloaded_songs[i]
            # page = 0
            # req_address = 'http://beatsaver.com/search/all/' + str(page * 40) + '?key=' + search_key
            # HTMLreq = Request(req_address, headers={'User-Agent': 'Mozilla/5.0'})
            # response = urlopen(HTMLreq)
            # HTML = str(response.read())
            # thumbsUpRegEx = re.compile("<i class=\"fas fa-thumbs-up\"><\/i>\s(.*?) \/")
            # thumbsDownRegEx = re.compile("<i class=\"fas fa-thumbs-down\"><\/i>\s(.*?)" + r"\\" + "n\s*?<\/li>")
            # fileNameMatches = re.findall(fileNameRegEx, HTML)  # Extract file names (both very hacky)
            # if len(fileNameMatches) > 0:
            #     match_found = True
            #     match_id = 0
            # else:
            #     print('No track found for: '+downloaded_songs_full[i])
            #     print('It is recommended that you delete '+downloaded_songs_full[i])

        if match_found is True:
            titleMatches = re.findall(titleRegEx, HTML)  # Extract song titles
            fileNameMatches = re.findall(fileNameRegEx, HTML)  # Extract file names (both very hacky)
            authorMatches = re.findall(authorRegEx, HTML)  # Extract author names
            downloadsMatches = re.findall(downloadsRegEx, HTML)  # Extract num downloads
            finishedMatches = re.findall(finishedRegEx, HTML)  # Extract num finished
            thumbsUpMatches = re.findall(thumbsUpRegEx, HTML)  # Extract thumb ups
            thumbsDownMatches = re.findall(thumbsDownRegEx, HTML)  # Extract thumb downs
            ratingMatches = re.findall(ratingRegEx, HTML)  # Extract rating

            # Write meta data text file
            if not os.path.exists(meta_data_txt_output):
                f = open(meta_data_txt_output, 'a').close()  # incase doesn't exist

            metasize = os.path.getsize(meta_data_txt_output)

            f = open(meta_data_txt_output, 'w')
            f.write('id: ' + html.unescape(fileNameMatches[match_id]) + ';\n')
            f.write('title: ' + html.unescape(titleMatches[match_id]) + ';\n')
            f.write('author: ' + html.unescape(authorMatches[match_id]) + ';\n')
            f.write('downloads: ' + html.unescape(downloadsMatches[match_id]) + ';\n')
            f.write('finished: ' + html.unescape(finishedMatches[match_id]) + ';\n')
            f.write('thumbsUp: ' + html.unescape(thumbsUpMatches[match_id]) + ';\n')
            f.write('thumbsDown: ' + html.unescape(thumbsDownMatches[match_id]) + ';\n')
            f.write('rating: ' + html.unescape(ratingMatches[match_id]) + ';\n')
            f.close()
            if metasize == 0:
                print('Created meta_data ' + os.path.join(downloaded_songs_full[i]))
            else:
                print('Updated meta_data for ' + downloaded_songs_full[i])

if __name__ == "__main__":
    #fileNamesA, titlesA = download_top_k_played_songs(10)
    update_meta_data_for_downloaded_songs()

