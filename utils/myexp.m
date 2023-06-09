function [f,logf,df,ddf] = myexp(x)
% [f,logf,df,ddf] = myexp(x)
%
% Replacement for 'exp' that returns 4 arguments: exp(x), log(exp(x)), and 1st & 2nd deriv

f = exp(x);
logf = x;
if nargout > 2
    df = f;
    ddf = f;
end
