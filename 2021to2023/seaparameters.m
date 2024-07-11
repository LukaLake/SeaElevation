clear all;
close all;

% 指定要读取的 NetCDF 文件的路径
ncFile = 'seaparameters.nc';

% 使用 ncinfo 函数获取文件的信息
fileInfo = ncinfo(ncFile);

% 获取文件中的变量名称
variableNames = {fileInfo.Variables.Name};

% 初始化一个空的结构体来存储所有变量的数据
data_struct = struct();

% 遍历每个变量并读取数据
for i = 1:numel(variableNames)
    % 使用 ncread 函数读取变量的数据
    data = ncread(ncFile, variableNames{i});
    
    if ndims(data) == 3
        % 获取数据的维度
        [rows, cols, depth] = size(data);
        
        % 初始化一个结构体，用来存储当前变量的每个深度数据
        currentVar_struct = struct();
        
        % 遍历深度维度，将每个矩阵存储到结构体中
        for j = 1:depth
            % 获取每个深度的数据矩阵
            currentMatrix = squeeze(data(:,:,j));
            
            % 使用索引作为标签
            matrixName = sprintf('matrix_%d', j);
            
            % 将矩阵存储到结构体中
            currentVar_struct.(matrixName) = currentMatrix(1:rows,1:rows);
        end
        
        % 将当前变量的结构体添加到总的数据结构体中
        data_struct.(variableNames{i}) = currentVar_struct;
    end
end

% 指出我们想要导出到 Excel 的变量及其新名称
varNamesToExport = {'zos', 'tob', 'sob','pbo'};
newNames = {'heightabovegeoid', 'temperature', 'salinity','pressureatseafloor'};

% 导出指定变量的数据到 Excel 文件
for v = 1:numel(varNamesToExport)
    % 检查此变量是否被读取并存储在 data_struct 中
    if isfield(data_struct, varNamesToExport{v})
        % 获取此变量的数据
        varData = data_struct.(varNamesToExport{v});
        
        % 对每个矩阵进行操作
        matrixNames = fieldnames(varData);
        for m = 1:numel(matrixNames)
            % 获取矩阵数据
            matrixData = varData.(matrixNames{m});
            
            % 定义导出文件的名称
            exportFileName = sprintf('%s_%d.csv', newNames{v}, m);
            
            % 将矩阵数据写入 Excel 文件
            writematrix(matrixData, exportFileName)
        end
    end
end
