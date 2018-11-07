#!/bin/bash
trainPath="/home/yangyuhao/data/road/data/test_data/hehe/jpg_data"
validationPath="/home/yangyuhao/data/road/data/test_data/hehe/test"
trainDirList=`ls $trainPath`

for DirName in $trainDirList
do
    cd $trainPath/$DirName
    dirNum=`ls -l|grep "^-"|wc -l`
    picList=`ls *.jpg`
    k=0
    for fileName in $picList
    do
        fileNameArr[k]=$fileName
        k=$k+1
    done

    arr=($(seq 1 $dirNum))
    num=${#arr[*]}
    # 需要转移到另外对应文件夹下的图片总数
    let filterNum=$num*1/4
    # 先随机生成一个指定范围的数字作为初始值
    res=${arr[$(($RANDOM%num))]}
    fileArr[1]=$res
    let i=2
    # 将所有生成的随机数保存进fileArr数组，作为要转移的图片的下标
    while(( i<=filterNum ));
    do
        res=${arr[$(($RANDOM%num))]}
        fileArr[i]=$res
        for((j=1;j<i;j++));
        do
        numJ=${fileArr[j]}
        if [[ $res == $numJ ]]; then
            unset fileArr[i]
            i=$i-1
            break
        fi
        done
        i=$i+1
    done

    cd $validationPath
    mkdir $DirName
    for((indexNum=0;indexNum<$filterNum;indexNum++))
    do
        mv $trainPath/$DirName/${fileNameArr[fileArr[indexNum]-1]} $validationPath/$DirName
    done
done