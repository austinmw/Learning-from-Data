%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ENG EC 503: Learning from Data
% Boston University,
% Instructor: Prakash Ishwar
% Assignment 5, Problem 5.1 Part (a)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Starting...');

load data_SFcrime_train.mat

% binarize DayOfWeek
DayOfWeek = lower(DayOfWeek); % convert to lower cases
Day_unique = unique(DayOfWeek); % get the unique elements
K_DayOfWeek = numel(Day_unique); % number of possible values
Day_bin = spalloc(numel(DayOfWeek), K_DayOfWeek, numel(DayOfWeek)); 
% binarize
Day_bin(:,1) = strcmp(DayOfWeek, 'sunday');
Day_bin(:,2) = strcmp(DayOfWeek, 'monday');
Day_bin(:,3) = strcmp(DayOfWeek, 'tuesday');
Day_bin(:,4) = strcmp(DayOfWeek, 'wednesday');
Day_bin(:,5) = strcmp(DayOfWeek, 'thursday');
Day_bin(:,6) = strcmp(DayOfWeek, 'friday');
Day_bin(:,7) = strcmp(DayOfWeek, 'saturday');

% binarize PdDistrict
PdDistrict = lower(PdDistrict);
Pd_unique = unique(PdDistrict);
K_Pd = numel(Pd_unique);
Pd_bin = spalloc(numel(PdDistrict), K_Pd, numel(PdDistrict));
for i = 1:K_Pd
    Pd_bin(:,i) = strcmp(PdDistrict,Pd_unique{i});
end

% binarize hour information
Hour = zeros(numel(Dates),1);
for i =1:numel(Dates)
    t = Dates{i};
    Hour(i) = str2double(t(end-4:end-3));
end
Hour_bin = spalloc(numel(Dates),24,numel(Dates));
for i=0:23
    Hour_bin(:,i+1) = (Hour == i);
end

%% most-likely hour for each crime
Crimes = unique(Category);
K_Crimes = numel(Crimes);
Crimes_Y = zeros(numel(Category),1);
mostlikely_hours = zeros(K_Crimes,1);
for i =1:K_Crimes
    temp = strcmp(Category,Crimes{i});
    Crimes_Y(temp)=i;
    % Find mostlikely hours
    hour_acc_temp = sum(Hour_bin(temp,:),1);
    [~,ind]=max(hour_acc_temp);
    mostlikely_hours(i)=ind-1;
end
clear hour_acc_temp ind i temp
%% Most-likely crime for each PD
mostlikely_crime_byPD = cell(K_Pd,1);
for i = 1:K_Pd
    temp = Pd_bin(:,i)==1;
    numtemp= mode(Crimes_Y(temp));
    mostlikely_crime_byPD{i}=Crimes{numtemp};
end
clear i numtemp temp
%% concat features
Features = [Day_bin, Pd_bin, Hour_bin];

disp('Finished.');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

