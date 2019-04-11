function result = runIndependent(K,lk)
% Original script by Nino Shervashidze, Karsten Borgwardt
% Modified by Asma Atamna
% K = kernel matrix (n*n)
% lk = vector of labels (n*1)
% cv = number of folds in cross-validation

addpath('libsvm-3.23/matlab/');
% randomly permute kernel matrix and labels
r = randperm(size(K,1));
K = K(r,r);
lk = lk(r);
lk=lk';
lkoriginal = lk; 
Koriginal = K;
% Split data into training (+ validation) data (90%)
% and test data (10%) that is fixed during the experiment.
p90_all = ceil(size(Koriginal,2) * 0.9);
lk = lkoriginal(1:p90_all,1); % Training labels
K = Koriginal(1:p90_all,1:p90_all); % Training data

% bring kernel matrix into libsvm format
p90 = ceil(size(K,2) * 0.9);
p100 = ceil(size(K,2) * 1.0);

% specify range of c-values
cvalues = (10 .^ [-7:2:7]) / size(K,2);

cv = 10;
fs = size(K,2) - p90;

% cross-validation loop
for k = 1:cv
    K = Koriginal(1:p90_all,1:p90_all);
    lk = lkoriginal(1:p90_all,1);

    K = K([k*fs+1:size(K,2),1:(k-1)*fs,(k-1)*fs+1:k*fs],[k*fs+1:size(K,2),1:(k-1)*fs,(k-1)*fs+1:k*fs]);  
    lk = lk([k*fs+1:size(K,2),1:(k-1)*fs,(k-1)*fs+1:k*fs]); 
    K = makepos(K);
    K1 = [(1:size(K,1))', normalizekm(K)];

    %if any(strcmp('optimal',options))
    imresult= zeros(1,size(cvalues,2));
    for i = 1:size(cvalues,2)
        % train on 80%, predict on 10% (from 81% to 90%)
        model = svmtrain(lk(1:p90,1), K1(1:p90,1:p90+1), strcat(['-t 4  -c ' num2str(cvalues(i))]));
        [predict_label, accuracy, dec_values] = svmpredict(lk(p90+1:p100,1),K1(p90+1:p100,1:p90+1), model);
        imresult(i) = imresult(i) + accuracy(1);
    end
end

% determine optimal c
Koriginal = makepos(Koriginal);
K1original = [(1:size(Koriginal,1))', normalizekm(Koriginal)];

[junk,optimalc]= max(fliplr(imresult));
optimalc = size(cvalues,2)+1 - optimalc; 
% train on 90% with optimal c, predict on 10% (from 91% to 100%)
model = svmtrain(lkoriginal(1:p90_all,1), K1original(1:p90_all,1:p90_all+1),strcat(['-t 4  -c ' num2str(cvalues(optimalc))]) );
[predict_label, accuracy, dec_values] = svmpredict(lkoriginal(p90_all+1:size(Koriginal,1),1), K1original(p90_all+1:size(Koriginal,1),1:p90_all+1), model);
result=accuracy(1)
end

function result = makepos(K)
pd = 0;
addc = 10e-7;
while (pd ==  0)
  
  addc = addc * 10
  try
    if (isinf(addc) == 1)
      pd = 1;
    else 
      chol(normalizekm(K + eye(size(K,1),size(K,1)) * addc));
      pd = 1;
    end
  catch
    
  end
  
end
if (isinf(addc)==0)
  result = K + eye(size(K,1),size(K,1)) * addc;
else
  result = eye(size(K,1));
end
end
