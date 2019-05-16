BEGIN
{
    # 出力ファイル名
    file_tc = "ftc.count10"
    file_t = "ftx.count10"
    file_c = "fxc.count10"
}

{
    # 文脈語がある物について連想配列でカウント
    if (NF != 1){
        t[$1]++
        for (i=2; i <= NF; i++){
            tc[($1"\t"$i)] + +
            c[$i]++
        }
    }

} END {
    # カウントが10以上のもののみ出力
    for ( key in tc )
        if (tc[key] >= 10)
            print tc[key] "\t" key > file_tc

    for ( key in t )
        if (t[key] >= 5)
            print t[key] "\t" key > file_t

    for ( key in c )
        if (c[key] >= 10)
            print c[key] "\t" key > file_c
}