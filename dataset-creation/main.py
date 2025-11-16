from dotenv import load_dotenv
from sqa_types import Transcript
from generate_questions import generate_qa
from index_files import load_transcript
from generate_tts import OptimizedTTS


base_dir = "/training-1/asr_dataset_files/asr_bundestag"
train = base_dir + "/train_nodev"
validate = base_dir + "/train_dev"
test = base_dir + "/test"
wav_dir = "/training-1/asr/bundestag/dataset/wavs"

load_dotenv(dotenv_path="../.env")

# first load transcripts
transcripts: list[Transcript] = load_transcript(train)

# initialize TTS Model
tts = OptimizedTTS(voice="Julian")

# generate questions + answers for transcripts using AI
for transcript in transcripts:
    questions = generate_qa(transcript['text'])
    print(questions)
    for question in questions:
        spokenQuestion = tts.generate(questions)
        # save to subdirectory transcripts/<transcript_id>/<difficulty>.wav

        # Generate TTS for the question
        spoken_question = tts.generate(question['question'])

        # Save to subdirectory transcripts/<transcript_id>/<difficulty>_<idx>.wav
        output_path = f"tts-audios/{question['difficulty']}.wav"

        # Save the audio file
        with open(output_path, 'wb') as f:
            f.write(spoken_question)
