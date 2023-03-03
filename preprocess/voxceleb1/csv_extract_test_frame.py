import glob, shutil, tqdm, os

dict_file = "/datasets2/voxceleb1/vox1_meta.csv"

# Read the meta file
lines = open(dict_file).read().splitlines()[1:]
# lines = list(filter(lambda x: x.split()[-1] == 'dev', lines))

id_name_dict = {}
for line in lines:
	speaker_id = line.split()[0]
	speaker_name = line.split()[1]
	id_name_dict[speaker_id] = speaker_name

text_files = glob.glob("/datasets2/voxceleb1/txt/*/*/*.txt")

t = 0
for text_file in tqdm.tqdm(text_files, total = len(text_files)):
	speaker_id = text_file.split('/')[-3]
	speaker_name = id_name_dict[speaker_id]
	video_id = text_file.split('/')[-2]
	utterance_id = text_file.split('/')[-1][:-4]
	cropped_folder = "/datasets2/voxceleb1/unzippedIntervalFaces/data/%s/1.6/%s/%d"%(speaker_name, video_id, int(utterance_id))
	target_folder = "/datasets2/voxceleb1/face/%s/%s/%s"%(speaker_id, video_id, utterance_id)
	try:
		out = shutil.copytree(cropped_folder, target_folder)
	except:
		folders = os.path.join('/'+cropped_folder.split('/')[0],cropped_folder.split('/')[1], \
      						   cropped_folder.split('/')[2],cropped_folder.split('/')[3], \
             				   cropped_folder.split('/')[4],cropped_folder.split('/')[5])  + '/1.6/*/*'
		cropped_folder = glob.glob(folders)[0]
		t += 1
		try:
			out = shutil.copytree(cropped_folder, target_folder)
		except:
			print(out)
print(t)