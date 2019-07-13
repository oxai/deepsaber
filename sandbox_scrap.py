import requests

from bs4 import BeautifulSoup
page=2
def get_soup(url):
    r = requests.get(url)
    return BeautifulSoup(r.content, "html.parser")

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(rank)

num_tasks = 12
num_tasks_per_job = num_tasks//size
tasks = list(range(rank*num_tasks_per_job,(rank+1)*num_tasks_per_job))
if rank < num_tasks%size:
    tasks.append(size*num_tasks_per_job+rank)


import json
# meta_data_filename = "metadata/BeatSaberScrappedData/combinedScrappedData.json"
# meta_data = json.load(open(meta_data_filename))
# meta_data = {x['Key']: x for x in meta_data}
# meta_data[beast_saber_ids[0]]
# page = 8
all_urls = []
for page in tasks:
    soup = get_soup('https://bsaber.com/curator-recommended/'+str(page))
    upvotes = list(map(lambda x: int(x.next.strip()), soup.select(".fa.fa-thumbs-up.fa-fw")))
    downvotes = list(map(lambda x: int(x.next.strip()), soup.select(".fa.fa-thumbs-down.fa-fw")))
    # song_names = list(map(lambda x: x.text.strip(), soup.select(".post-title.entry-header")))
    # authors = list(map(lambda x: x.text.strip(), soup.select(".mapper_id.vcard")))
    # beat_saber_ids = list(map(lambda x: x.select_one("a")["href"].split("/")[-2], soup.select(".post-title.entry-header")))
    urls = list(map(lambda x: x["href"],soup.select(".action.post-icon.bsaber-tooltip.-download-zip")))
    beast_saber_ids = list(map(lambda x: x.split("/")[-1], urls))
    for i, url in enumerate(urls):
        print(beast_saber_ids[i])
        r = requests.get(url, allow_redirects=True)
        open("Data/"+beast_saber_ids[i]+".zip", 'wb').write(r.content)
    # all_urls.append(list(urls))

# all_urls = comm.gather(all_urls, root=0)
# if rank == 0:
#     print(all_urls)

# import imp; import DownloadData; imp.reload(DownloadData)
# from DownloadData import get_scoresaber_id_of_song, get_beastsaber_meta_from_id, get_scoresaber_difficulty_from_scoresaber_id
#
# get_scoresaber_id_of_song("We Will Rock You (2011 Remaster) - Queen", "Joetastic")
