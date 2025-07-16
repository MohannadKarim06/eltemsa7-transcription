import yt_dlp

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
    'outtmpl': '/content/drive/MyDrive/Untitled Folder/downloaded_audio/%(title)s.%(ext)s', # This line specifies the output directory

}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=upgAxjEZ7YU&list=PLfeJT8wCesumA5kQvGS4SsFVopuuFAFQ8'])
