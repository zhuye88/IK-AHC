function eva_setup = setup()
    root = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(root, 'eva')));
    addpath(genpath(fullfile(root, 'src')));
    addpath(genpath(fullfile(root, 'demo')));
end