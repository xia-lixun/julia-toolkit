function locate(path::String, filetype::String)    
    
    result = Array{String}(0)
    list = readdir(path)    
    for i in list
        if isdir("$path$i")
            append!(result, locate("$path$i\\", filetype) )
        elseif isfile("$path$i")
            if i[end-3:end] == filetype
                push!(result, "$path$i")
            end
        end
    end
    return result
end