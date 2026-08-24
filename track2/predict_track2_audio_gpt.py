import os, json, base64
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

DATA_DIR  = '/data/multimer/ACMMM2026/dataset/mer2026-dataset'
AUDIO_DIR = '/data/multimer/ACMMM2026/dataset/mer2026-dataset-process/audio'
OUT_DIR   = '/data/multimer/ACMMM2026/results/track2'
os.makedirs(OUT_DIR, exist_ok=True)

MODEL = 'gpt-audio'

SYSTEM_MSG = ("You are an expert in multimodal fine-grained emotion analysis. "
              "Output only comma-separated emotion words, no explanation.")

PROMPT = """Audio clip: (see audio attachment)
Subtitle (Chinese): {subtitle}

Examples:
- "I can't believe you did this" → shocked,betrayed,hurt,angry,disappointed
- "Congratulations on the promotion!" → happy,excited,proud,grateful
- "We need to talk about what happened" → worried,nervous,anxious,apprehensive

Predict the emotions expressed in the audio. Output ONLY 3-6 comma-separated emotion words (lowercase):"""

def load_audio_b64(name):
    path = os.path.join(AUDIO_DIR, f'{name}.wav')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def predict_one(name, subtitle, client):
    audio_b64 = load_audio_b64(name)
    content = []
    if audio_b64:
        content.append({'type': 'input_audio', 'input_audio': {'data': audio_b64, 'format': 'wav'}})
    content.append({'type': 'text', 'text': PROMPT.format(subtitle=subtitle[:400])})

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            modalities=['text'],
            messages=[
                {'role': 'system', 'content': SYSTEM_MSG},
                {'role': 'user',   'content': content},
            ],
            max_completion_tokens=60,
            timeout=60,
        )
        pred = resp.choices[0].message.content.strip().lower()
        emotions = [e.strip() for e in pred.replace('\n', ',').split(',') if e.strip()]
        emotions = [e for e in emotions if e and len(e) < 30][:6]
        return name, ','.join(emotions) if emotions else 'neutral'
    except Exception as e:
        print(f'Error {name}: {e}')
        return name, 'neutral'

def run(max_workers=8, resume=True, out_suffix='gpt_audio'):
    client = OpenAI()
    df_cand = pd.read_csv(f'{DATA_DIR}/track1_track2_candidate.csv')
    df_sub  = pd.read_csv(f'{DATA_DIR}/subtitle_chieng.csv').set_index('name')

    out_path = f'{OUT_DIR}/predictions_{out_suffix}.jsonl'
    done = {}
    if resume and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line); done[r['name']] = r['prediction']
                except: pass
        print(f'Resuming: {len(done)} done')

    todo = df_cand[~df_cand['name'].isin(done)]
    print(f'Predicting {len(todo)} samples with {MODEL}, workers={max_workers}...')

    with open(out_path, 'a') as fout:
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = {}
            for _, row in todo.iterrows():
                name = row['name']
                subtitle = str(df_sub.loc[name, 'chinese']) if name in df_sub.index else ''
                if subtitle == 'nan': subtitle = ''
                futures[exe.submit(predict_one, name, subtitle, client)] = name
            for fut in tqdm(as_completed(futures), total=len(futures)):
                name, pred = fut.result()
                done[name] = pred
                fout.write(json.dumps({'name': name, 'prediction': pred}) + '\n')
                fout.flush()

    df_cand['prediction'] = df_cand['name'].map(done).fillna('neutral')
    out_csv = f'{OUT_DIR}/answer_{out_suffix}.csv'
    df_cand[['name', 'prediction']].to_csv(out_csv, index=False)
    print(f'Saved {len(df_cand)} → {out_csv}')
    import zipfile
    with zipfile.ZipFile(out_csv.replace('.csv', '.zip'), 'w') as z:
        z.write(out_csv, 'answer.csv')
    print(f'Submission: {out_csv.replace(".csv", ".zip")}')

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--workers',    type=int, default=8)
    p.add_argument('--out_suffix', type=str, default='gpt_audio')
    p.add_argument('--no_resume',  action='store_true')
    args = p.parse_args()
    run(args.workers, not args.no_resume, args.out_suffix)
