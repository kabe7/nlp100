import re
from pathlib import Path
from random import randint
import numpy as np
from collections import Counter
from itertools import accumulate
import scipy.sparse as spar
from sklearn.decomposition import TruncatedSVD

files = {
    'raw_data': Path.home() / 'tmp' / '09' / "enwiki-20150112-400-r10-105752.txt",
    'formatted_data': Path.home() / 'tmp' / '09' / 'formatted.txt',
    'corpus': Path.home() / 'tmp' / '09' / 'corpus.txt',
    'country_names': Path.cwd() / '..' / 'data' / 'cname.txt',
    'word_list': Path.cwd() / '..' / 'data' / 'ftc' / 'word.sort',
    'ftc': Path.cwd() / '..' / 'data' / 'ftc' / 'ftc.sort',
    'ftx': Path.cwd() / '..' / 'data' / 'ftc' / 'ftx.sort',
    'fxc': Path.cwd() / '..' / 'data' / 'ftc' / 'fxc.sort',
    'word_vec': Path.cwd() / '..' / 'out' / 'wvec.npz',
    'word_vec_300': Path.cwd() / '..' / 'out' / 'wvec300.npy',
}

print('# 80: コーパスの制定')
print('# 81: 複合語からなる国名への対処')

if not files['formatted_data'].exists():
    # 複合語の国の置換関数を作成
    with files['country_names'].open(encoding='utf-8') as cname:
        dic = {line.strip(): line.strip().replace(' ', '_') for line in cname}
        translator = re.compile('|'.join(dic.keys()))
        add_underbar = lambda match: dic[match.group(0)]

    # データの読み込み/コーパスの書き出し
    with \
            files['raw_data'].open(encoding='utf-8') as src, \
            files['formatted_data'].open('w', encoding='utf-8') as res:

        for line in src:
            # 80: コーパスの制定
            words = [re.sub(r'\A[.,!?;:()\[\]\'\"]+|[.,!?;:()[\]"\']+\Z', '', word) for word in line.split()]
            text = ' '.join(word for word in words if word)
            # 81: 複合語からなる国名への対処
            res.write(translator.sub(add_underbar, text) + '\n')


print('# 82. 文脈の抽出')
if not files['corpus'].exists():
    with \
            files['formatted_data'].open(encoding='utf-8') as src, \
            files['corpus'].open('w', encoding='utf-8') as res:
        for line in src:
            words = line.split()
            n = len(words)
            for idx, word in enumerate(words):
                d = randint(1,5)
                start, end = max(0, idx-d), min(n,idx+d+1)
                context = words[start:end]
                context.remove(word)
                res.write(word+'\t'+(' '.join(context))+'\n')

print('# 83. 単語／文脈の頻度の計測')

print('# 84. 単語文脈行列の作成')
if not files['word_vec'].exists():
    # 単語とindexの対応づけ
    with files['word_list'].open(encoding='utf-8') as wl:
        w2num = {word.strip(): idx for idx, word in enumerate(wl)}

    # ppmiの計算に必要な数値を単語と関連づけておく
    with files['ftx'].open(encoding='utf-8') as ts:
        log_ftx = {word: np.log(int(num)) for num, word in (line.split() for line in ts) if word in w2num}

    with files['fxc'].open(encoding='utf-8') as cs:
        log_fxc = {word: np.log(int(num)) for num, word in (line.split() for line in cs) if word in w2num}

    N = 6894484568; log_N = np.log(N)

    # 共起の組み合わせから疎行列を構成していく
    # data: 非ゼロ要素の値
    # indices: 非ゼロ要素の列番号
    # indptr: dataのどの区間が南郷目にあたるかの情報を持つ
    with files['ftc'].open(encoding='utf-8') as tcs:
        data, indices, indptr_dic = [], [], Counter()
        for line in tcs:
            _fct_, _t, _c = line.split()
            fct, t, c = int(_fct_), _t, _c.strip()
            try:
                ppmi = np.log(fct) + log_N - log_ftx[t] - log_fxc[c]
                if ppmi > 0:
                    data.append(ppmi)
                    indices.append(w2num[c])
                    indptr_dic[t]+=1

            except KeyError:
                pass

    indptr = [0]
    indptr.extend(
        accumulate(
            indptr_dic[word] for word in (word for word, _ in sorted(w2num.items(), key=lambda t:t[1])
                                      )
        )
    )
    n = len(indptr)-1
    res = spar.csr_matrix((np.array(data), np.array(indices), np.array(indptr)), (n,n))
    spar.save_npz(files['word_vec'], res)

print('# 85. 主成分分析による次元圧縮')
if not files['word_vec_300'].exists():
    wvec = spar.load_npz(files['word_vec'])
    pca = TruncatedSVD(n_components=300)
    wvec300 = pca.fit_transform(wvec)
    np.save(files['word_vec_300'], wvec300)

print('# 86. 単語ベクトルの表示')
if False:
    keyword = "United_States"
    with files['word_list'].open(encoding='utf-8') as wl:
        for n, word in enumerate(wl):
            if word.strip() == keyword:
                break

    wv = np.load(files['word_vec_300'])
    #print(wv[n])

print('# 87. 単語の類似度')
if True:
    keywords = ["United_States", "U.S"]
    dic = {}
    with files['word_list'].open(encoding='utf-8') as wl:
        for n, word in enumerate(wl):
            if word.strip() in keywords:
                dic[word.strip()] = n

    wv = np.load(files['word_vec_300'])
    x, y = [wv[dic[word]] for word in keywords]
    print("similarity between United_States and U.S is ", x@y / (np.linalg.norm(x) * np.linalg.norm(y)))

print('# 88. 類似度の高い単語10件')
if True:
    keyword = "England"
    with files['word_list'].open(encoding='utf-8') as wl:
        for n, word in enumerate(wl):
            if word.strip() == keyword:
                break

    wv = np.load(files['word_vec_300'])
    key_vec = wv[n]
    cos_array = wv@key_vec / (np.linalg.norm(wv,axis=1) * np.linalg.norm(key_vec))
    top10 = np.argsort(-cos_array)[:10]
    with files['word_list'].open(encoding='utf-8') as wl:
        for n, word in enumerate(wl):
            if n in top10:
                print(word, cos_array[n])

print('# 89. 加法構成性によるアナロジー')
if True:
    words = sp, md, at = "Spain", "Madrid", "Athens"
    dic = {}
    with files['word_list'].open(encoding='utf-8') as wl:
        for n, word in enumerate(wl):
            if word.strip() in words:
                dic[word.strip()] = n

    wv = np.load(files['word_vec_300'])
    sp_vec, md_vec, at_vec = [wv[dic[word]] for word in [sp, md, at]]
    key_vec = sp_vec - md_vec + at_vec
    cos_array = wv @ key_vec / (np.linalg.norm(wv, axis=1) * np.linalg.norm(key_vec))
    top10 = np.argsort(-cos_array)[:10]
    with files['word_list'].open(encoding='utf-8') as wl:
        for n, word in enumerate(wl):
            if n in top10:
                print(word, cos_array[n])