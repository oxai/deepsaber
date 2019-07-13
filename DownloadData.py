from urllib.request import Request, urlopen
import re
import zipfile
import pickle
import os
import html

from IOFunctions import write_meta_data_file, read_meta_data_file

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, 'Data')
EXTRACT_DIR = os.path.join(THIS_DIR, 'DataE')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)

def download_top_k_played_songs(k, update_existing=False):
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
    while total_downloaded < k:
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
            if total_downloaded < k:
                fileName = re.sub(illegalNameChars, "", html.unescape(titleMatches[index])).rstrip(' - ')

                if fileName not in downloaded_songs:
                    new_song = True
                    song_name = str(total_downloaded+1)+")"+fileName
                    dataReq = Request('http://beatsaver.com/download/'+match, headers={'User-Agent': 'Mozilla/5.0'})
                    data = urlopen(dataReq).read()
                    try:
                        pickle.dump(data, open(os.path.join(DATA_DIR, song_name+".zip"), "wb"))  # Store file
                    except:
                        song_name = str(total_downloaded+1)+')'
                        pickle.dump(data, open(os.path.join(DATA_DIR, song_name + ").zip"), "wb"))  # Safe file name choice
                    # Extract the zipfile contents to other folder
                    os.mkdir(os.path.join(EXTRACT_DIR, song_name))
                    zip_ref = zipfile.ZipFile(os.path.join(DATA_DIR, song_name+".zip"), 'r')
                    zip_ref.extractall(os.path.join(EXTRACT_DIR, song_name))  # Need to create new folder DataE to work
                    zip_ref.close()
                    print('Extracted song ' + str(total_downloaded) + ' of ' + str(k) + ': ' + song_name)
                    total_downloaded += 1
                    nbDownloaded += 1
                else:
                    new_song = False
                    song_name = downloaded_songs_full[downloaded_songs.index(fileName)]
                    print('Confirmed Song ' + song_name)

                if new_song is True or update_existing is True:
                    # Write meta data text file
                    meta_data_txt_output = os.path.join(os.path.join(EXTRACT_DIR, song_name),
                                                        'meta_data.txt')
                    meta_data = dict()
                    meta_data['id'] = html.unescape(fileNameMatches[index])
                    meta_data['title'] = html.unescape(titleMatches[index])
                    meta_data['author'] = html.unescape(authorMatches[index])
                    meta_data['downloads'] = html.unescape(downloadsMatches[index])
                    meta_data['finished'] = html.unescape(finishedMatches[index])
                    meta_data['thumbsUp'] = html.unescape(thumbsUpMatches[index])
                    meta_data['thumbsDown'] = html.unescape(thumbsDownMatches[index])
                    meta_data['rating'] = html.unescape(ratingMatches[index])
                    meta_data['scoresaberId'] = get_scoresaber_id_of_song(meta_data['title'], meta_data['author'])
                    meta_data['scoresaberDifficulty'], meta_data['scoresaberDifficultyLabel'] = get_scoresaber_difficulty_from_scoresaber_id(meta_data['scoresaberId'])
                    meta_data.update(get_beastsaber_meta_from_id(meta_data['id']))
                    write_meta_data_file(meta_data_txt_output, meta_data)
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
            meta_data = read_meta_data_file(meta_data_txt_output)
        else:
            meta_data = dict()
            meta_data['id'] = None
            meta_data['author'] = None
            search_key = None

        match_found = False
        if meta_data['id'] is not None:
            req_address = 'http://beatsaver.com/browse/detail/'+meta_data['id']
            HTMLreq = Request(req_address, headers={'User-Agent': 'Mozilla/5.0'})
            response = urlopen(HTMLreq)
            HTML = str(response.read())
            thumbsUpRegEx = re.compile("<span>Up\s(.*?)<\/span>")
            thumbsDownRegEx = re.compile("<span>Down\s(.*?)<\/span>")
            match_found = True
            match_id = 0
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

            meta_data['id'] = html.unescape(fileNameMatches[match_id])
            meta_data['title'] = html.unescape(titleMatches[match_id])
            meta_data['author'] = html.unescape(authorMatches[match_id])
            meta_data['downloads'] = html.unescape(downloadsMatches[match_id])
            meta_data['finished'] = html.unescape(finishedMatches[match_id])
            meta_data['thumbsUp'] = html.unescape(thumbsUpMatches[match_id])
            meta_data['thumbsDown'] = html.unescape(thumbsDownMatches[match_id])
            meta_data['rating'] = html.unescape(ratingMatches[match_id])
            meta_data['scoresaberId'] = get_scoresaber_id_of_song(meta_data['title'], meta_data['author'])
            meta_data['scoresaberDifficulty'], meta_data['scoresaberDifficultyLabel'] = get_scoresaber_difficulty_from_scoresaber_id(meta_data['scoresaberId'])
            meta_data.extend(get_beastsaber_meta_from_id(meta_data['id']))
            write_meta_data_file(meta_data_txt_output, meta_data)
            print('Saved meta_data for ' + downloaded_songs_full[i])


def get_scoresaber_id_of_song(song_identifier, author_identifier):
    # Get the HTML page. The page has 20 files on it
    HTMLreq = Request('https://scoresaber.com/?search=' + author_identifier, headers={'User-Agent': 'Mozilla/5.0'})
    response = urlopen(HTMLreq)
    HTML = str(response.read().decode())
    print(HTML)
    titleRegEx = re.compile("<td class=\"song\">\\r\\n\s*?<img src=\"[\/a-zA-Z0-9]*.png\" \/>\\r\\n\s*<a href=\"\/leaderboard\/[0-9]*\">\\r\\n\s*(.*?(?=\\r))")
    leaderboardURLRegEx = re.compile("<a href=\"(\/leaderboard\/[0-9]*)\">")
    illegalNameChars = re.compile("[\/\\:\*\?\"<>|]")
    titleMatches = re.findall(titleRegEx, HTML)  # Extract song title
    leaderboardMatches = re.findall(leaderboardURLRegEx, HTML)  # Extract song title
    leaderboard_id = []
    for index, match in enumerate(titleMatches):
        # Download the corresponding ZIP file
        fileName = re.sub(illegalNameChars, "", html.unescape(titleMatches[index])).strip()
        if fileName == song_identifier:
            leaderboard_id.append(int(leaderboardMatches[index].split('/')[-1]))
    if len(leaderboard_id) > 0:
        print('By jove we\'ve found them!')
    print(str(len(leaderboard_id)) + ' scoresaber entries were found matching song: ' + song_identifier +
          ', author: ' + author_identifier + ';')
    return leaderboard_id


def get_scoresaber_difficulty_from_scoresaber_id(scoresaber_id):
    try:
        num_songs = len(scoresaber_id)
    except TypeError:
        scoresaber_id = [scoresaber_id]
        num_songs = len(scoresaber_id)
    difficulty = []
    difficultyLabel = []
    print('Searching '+str(num_songs)+' scoresaber entries for difficulty rating.')
    for id in scoresaber_id:
        HTMLreq = Request('https://scoresaber.com/leaderboard/'+str(id), headers={'User-Agent': 'Mozilla/5.0'})
        response = urlopen(HTMLreq)
        HTML = str(response.read().decode())
        difficultyRegEx = re.compile("Star Difficulty: <b>([0-9]*\.[0-9]*)")
        difficultyLabelRegEx = re.compile("<h4 class=\"title is-5\" style=\"margin-top:50px\">[a-zA-Z0-9 \W]*\(<span style=\"color:\#?[a-zA-Z0-9]*;\">([a-zA-Z]*)<\/span>\)")
        difficultyMatches = re.findall(difficultyRegEx, HTML)  # Extract song title
        difficultyLabelMatches = re.findall(difficultyLabelRegEx, HTML)  # Extract song title
        if len(difficultyMatches) == 1:
            difficulty.append(float(difficultyMatches[0]))
            if len(difficultyLabelMatches) == 1:
                difficultyLabel.append(difficultyLabelMatches[0])
                print('Leaderboard entry ' + str(id) + ' has a difficulty of ' + str(
                    difficultyMatches[0]) + ' and a label of ' + str(difficultyLabelMatches[0]))
            else:
                difficultyLabel.append(None)
                print('Leaderboard entry ' + str(id) + ' has a difficulty of ' + str(
                    difficultyMatches[0]))
        else:
            difficulty.append(None)
            difficultyLabel.append(None)
            # print('Scoresaber difficulty not found for leaderboard entry ' + str(id))
    return difficulty, difficultyLabel

def get_beastsaber_meta_from_id(song_id):
    beastsaber_meta = dict()
    beastsaber_id = str(song_id).split('-')[0]
    HTMLreq = Request('https://bsaber.com/songs/'+str(beastsaber_id)+'/', headers={'User-Agent': 'Mozilla/5.0'})
    response = urlopen(HTMLreq)
    HTML = str(response.read().decode())
    criterionScoreRegEx = re.compile("<span class=\"rwp-criterion-score\" style=\"line-height: 18px; font-size: 18px;\">([0-9]*.[0-9]*)<\/span>")
    criterionLabelRegEx = re.compile("<span class=\"rwp-criterion-label\" style=\"line-height: 14px;\">([a-zA-Z ]*)<\/span>")
    criterionScoreMatches = re.findall(criterionScoreRegEx, HTML)  # Extract song title
    criterionLabelMatches = re.findall(criterionLabelRegEx, HTML)  # Extract song title
    if len(criterionScoreMatches) == 6 and len(criterionLabelMatches) == 6:
        print('Beastsaber stats extracted successfully: ')
        criterionLabelMatches = [x[0].lower() + x[1:] for x in criterionLabelMatches]
        for i in range(6):
            criterionLabelMatches = [x[0].lower() + x[1:].replace(" ", "") for x in criterionLabelMatches]
            beastsaber_meta.update({criterionLabelMatches[i]: criterionScoreMatches[i]})
        for key, value in beastsaber_meta.items():
            print(str(key)+': '+str(value))
    return beastsaber_meta

if __name__ == "__main__":
    fileNamesA, titlesA = download_top_k_played_songs(1000, update_existing=False)
    #update_meta_data_for_downloaded_songs()
