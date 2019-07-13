const express = require('express')
const app = express()
const port = 3000

app.get('*', (req, res) => {
  // res.send('Hello World!')
  console.log(req.url)
  download_song(req.url.slice(1)).then(()=>{
    generate_ddc().then((thing)=> {
      console.log(thing);
      let folder_path = thing.split(".")[0];
      let sm_filename = folder_path.split("/").slice(-1)[0]
      let sm_path = folder_path + "/" + sm_filename + ".sm"
      generate_deepsaber(res,sm_path)
    })
  })
})

const util = require('util');
const exec = util.promisify(require('child_process').exec);

// async function generate() {
//   const { stdout, stderr } = await exec('cd ../base; ./script_generate.sh ../../smolsong.wav');
//   console.log('stdout:', stdout);
//   console.log('stderr:', stderr);
// }

const { spawn } = require('child_process');

function download_song(url) {
  return new Promise((resolve, reject) => {
    console.log("Downloading song from youtube");
    exec('ytdl '+url+' | ffmpeg -y -i pipe:0 -b:a 192K -vn /home/guillefix/songs/song.mp3', (error, stdout, stderr) => {
      // console.log('stdout:', stdout);
      // console.log('stderr:', stderr);
      resolve(0)
    })
  })
}

function generate_ddc() {

  return new Promise((resolve, reject) => {

    console.log("DDC Stage 1");

    const child = spawn('python2', ['ddc_client.py','song', 'song', '/home/guillefix/songs/song.mp3'], {cwd: "../../ddc/infer"});

    child.stdout.setEncoding('utf8').on('data', (chunk) => {
      // let url = chunk.split("\n").slice(-2)[0]
      resolve(chunk)
    });
    child.stderr.setEncoding('utf8').on('data', (chunk) => {
      // let url = chunk.split("\n").slice(-2)[0]
      console.log(chunk);
      // resolve(chunk)
    });
    child.on('close', (code) => {
      console.log(`child process exited with code ${code}`);
    });

  })

}

async function generate_deepsaber(res,sm_path) {

  console.log("Deepsaber");
  console.log("sm_path",sm_path);

  const child = spawn('./script_generate.sh', ['/home/guillefix/songs/song.mp3', sm_path], {cwd: "../base"});

  // use child.stdout.setEncoding('utf8'); if you want text chunks
  child.stdout.setEncoding('utf8').on('data', (chunk) => {
    // console.log(chunk)
    let url = chunk.split("\n").slice(-2)[0]
    console.log(url);
    res.send(url)
    // data from standard output is here as buffers
  });

  child.stderr.setEncoding('utf8').on('data', (chunk) => {
    console.log(chunk)
  });

  // since these are streams, you can pipe them elsewhere
  // child.stderr.pipe(dest);

  child.on('close', (code) => {
    console.log(`child process exited with code ${code}`);
  });


}

app.listen(port, () => console.log(`Example app listening on port ${port}!`))
