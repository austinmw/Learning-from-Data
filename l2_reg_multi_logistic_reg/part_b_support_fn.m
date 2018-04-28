%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ENG EC 503: Learning from Data
% Boston University,
% Instructor: Prakash Ishwar
% Assignment 5, Problem 5.1
% Supporting function for part (b) to pre-process the raw data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% binarize DayOfWeek
DayOfWeek = lower(DayOfWeek); % convert to lower cases
Day_unique = unique(DayOfWeek); % get the unique elements
K_DayOfWeek = numel(Day_unique); % number of possible values
Day_bin = spalloc(numel(DayOfWeek), K_DayOfWeek, numel(DayOfWeek)); 

% binarize
for i = 1:K_DayOfWeek
    Day_bin(:,i) = strcmp(DayOfWeek,Day_unique{i});
end
clear i
%% binarize PdDistrict
PdDistrict = lower(PdDistrict);
Pd_unique = unique(PdDistrict);
K_Pd = numel(Pd_unique);
Pd_bin = spalloc(numel(PdDistrict), K_Pd, numel(PdDistrict));
for i = 1:K_Pd
    Pd_bin(:,i) = strcmp(PdDistrict,Pd_unique{i});
end
clear i
%% binarize hour information
Hour = zeros(numel(Dates),1);
for i =1:numel(Dates)
    t = Dates{i};
    Hour(i) = str2double(t(end-4:end-3));
end
clear t
Hour_bin = spalloc(numel(Dates),24,numel(Dates));
for i=0:23
    Hour_bin(:,i+1) = (Hour == i);
end
clear t i
%% numerize Crimes class label
Crimes = unique(Category);
K_Crimes = numel(Crimes);
Crimes_Y = zeros(numel(Category),1);
for i =1:K_Crimes
    temp = strcmp(Category,Crimes{i});
    Crimes_Y(temp)=i;
    % Find mostlikely hours
end

clear i temp

%% concat features
Features = [Day_bin, Pd_bin, Hour_bin];
clear X Y
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 