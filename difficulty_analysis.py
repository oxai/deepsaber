from urllib.request import Request, urlopen
import os
import re

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, 'Data')
EXTRACT_DIR = os.path.join(THIS_DIR, 'DataE')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)

''' @TS: Trying to regress a model of difficulty from beatsaber level features:
x_1) Number of blocks
x_2) Number of unique blocks
x_3) Number of obstacles
x_4) Number of bombs
x_5) Beats per minute
x_6) Song length


_Version 1_
Linear model: y = a0 + a1*x1 + a2*x2 + a3*x3 + a4*x4 + a5*x5
'''

#def generate_difficulty_features():
    # Generating Data Frame
    # Step 1 - Load json file
    # Read:     Number of blocks
    #           Number of unique blocks
    #           Number of obstacles
    #           Number of bombs

    # Step 2 - Load ogg file
    # Read:     Beats per minute
    #           Song length

    # Step 3 - Load difficulty rating
    # Read:     Difficulty Rating
    #           Number of votes

def get_difficulty_from_scoresaber(song_identifier, author_identifier):
    # Get the HTML page. The page has 20 files on it
    HTMLreq = Request('https://scoresaber.com/?search=' + author_identifier, headers={'User-Agent': 'Mozilla/5.0'})
    response = urlopen(HTMLreq)
    HTML = str(response.read().decode())
    titleRegEx = re.compile("<td class=\"song\">\\r\\n\s*?<img src=\"[\/a-zA-Z0-9]*.png\" \/>\\r\\n\s*<a href=\"\/leaderboard\/[0-9]*\">\\r\\n\s*(.*?(?=\\r))")
    illegalNameChars = re.compile("[\/\\:\*\?\"<>|]")
    titleMatches = re.findall(titleRegEx, HTML)  # Extract song titles
    for index, match in enumerate(titleMatches):
        # Download the corresponding ZIP file
        fileName = re.sub(illegalNameChars, "", titleMatches[index])
        if fileName == song_identifier:
            print('By jove we\'ve found him!')
    return None


def get_list_of_downloaded_songs():
    downloaded_songs_full = [dI for dI in os.listdir(EXTRACT_DIR) if os.path.isdir(os.path.join(EXTRACT_DIR, dI))]
    i = 0
    downloaded_songs = []
    while i < len(downloaded_songs_full):
        try:
            downloaded_songs.append(downloaded_songs_full[i].split(')', 1))
        except:
            print('Fail @ ' + str(i) + ': ' + downloaded_songs_full[i])
            pass
        i += 1
    return downloaded_songs_full, downloaded_songs

def read_meta_data_file(filename):
    num_lines = sum(1 for line in open(filename))
    f = open(meta_data_filename, 'r')
    meta_data = {}
    if num_lines > 6:
        meta_data['id'] = f.readline().split(': ')[1].split('\n')[0]
        if num_lines > 7:
            meta_data['title'] = f.readline().split(': ')[1].split('\n')[0]
    meta_data['author'] = f.readline().split(': ')[1].split('\n')[0]
    meta_data['downloads'] = f.readline().split(': ')[1].split('\n')[0]
    meta_data['finished'] = f.readline().split(': ')[1].split('\n')[0]
    meta_data['thumbsUp'] = f.readline().split(': ')[1].split('\n')[0]
    meta_data['thumbsDown'] = f.readline().split(': ')[1].split('\n')[0]
    try:
        meta_data['rating'] = f.readline().split(': ')[1].split('\n')[0]
    except:
        print('fail')
    return meta_data

if __name__ == '__main__':
    downloaded_songs_full, downloaded_songs = get_list_of_downloaded_songs()
    for song_dir in downloaded_songs_full:
        meta_data_filename = os.path.join(EXTRACT_DIR, os.path.join(song_dir, 'meta_data.txt'))
        if not os.path.exists(meta_data_filename):
            pass
        else:
            meta_data = read_meta_data_file(meta_data_filename)
            difficulty = get_difficulty_from_scoresaber(meta_data['title'], meta_data['author'])
            if difficulty == None:
                pass
