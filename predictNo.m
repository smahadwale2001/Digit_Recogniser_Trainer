function [no] = predictNo(result)
[a,no] = max(result,[],1);
if(no==10)
    no=0;
end
end