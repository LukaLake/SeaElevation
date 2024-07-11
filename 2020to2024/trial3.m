clear all;
close all;
% 指定要读取的 NetCDF 文件的路径
ncFile = '2020to2024.nc';

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

        % 检查数据是否有37个矩阵，如果是，则按照月份进行分割
        if depth == 42
            % 初始化一个结构体，用来存储当前变量的数据
            currentVar_struct = struct();

            for j = 1:depth
                % 获取每个月的数据并将其保存到对应的月份内的结构体
                currentMatrix = squeeze(data(:,:,j));
                
                % 计算矩阵非 NAN 元素的均值
                nonNanElements = currentMatrix(~isnan(currentMatrix));
                meanVal = mean(nonNanElements);
                
                % 使用均值替换矩阵中的 NAN 元素
                currentMatrix(isnan(currentMatrix)) = meanVal;
                
                currentVar_struct.(['month' num2str(j)]) = currentMatrix;
            end

            % 将当前变量的结构体添加到总的数据结构体中
            data_struct.(variableNames{i}) = currentVar_struct;
        end
    end
end

% 你可以通过 data_struct.variableName.monthNumber 来获取每一个月份对应变量的数据，其中NaN已用均值进行了替换。
% 先前的所有代码...
  
% 先前的所有代码...
  
% 指出我们想要导出到 Excel 的变量
varNamesToExport = {'zos', 'tob', 'sob'};

for v = 1:numel(varNamesToExport)
    % 检查此变量是否被读取并存储在 data_struct 中
    if isfield(data_struct, varNamesToExport{v})
        % 获取此变量的数据
        varData = data_struct.(varNamesToExport{v});
        
        % 对每个月的数据进行操作
        for m = 1:numel(fieldnames(varData))
            % 获取矩阵数据
            matrixData = varData.(['month' num2str(m)]);
            
            % 定义导出文件的名称
            switch varNamesToExport{v}
                % 根据变量名为导出文件命名
                case 'zos'
                    exportFileName = sprintf('height_month%d.xlsx', m);
                case 'tob'
                    exportFileName = sprintf('temperature_month%d.xlsx', m);
                case 'sob'
                    exportFileName = sprintf('salinity_month%d.xlsx', m);
            end
            
            % 将矩阵数据写入 Excel 文件
            writematrix(matrixData, exportFileName)
        end
    end
end