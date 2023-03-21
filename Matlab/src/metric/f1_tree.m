function [Bf1_score, Bcl]= f1_tree(Z, class, class_num)

    f1_score = [];

    for cl = class_num
        idx = cluster(Z, 'maxclust', cl);
        f1 = Fmeasure(class, idx);
        f1_score = [f1_score  f1];
    end
    [Bf1_score,j]=max(f1_score);
    Bcl=class_num(j);
end
