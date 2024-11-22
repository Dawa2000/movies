package column_selection;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class MovieMapper extends Mapper<LongWritable, Text, LongWritable, Text> {
    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        // Skip the header row
        if (key.get() == 0 && value.toString().contains("id")) {
            return;
        }

        // Split the CSV line
        String[] columns = value.toString().split(",");

        // Check if the row has the expected number of columns to avoid
        // ArrayIndexOutOfBoundsException
        if (columns.length < 13) {
            return; // Skip malformed rows
        }

        // Extract the required columns based on their indices
        String[] requiredColumns = {
                columns[1], // title
                columns[2], // vote_average
                columns[3], // vote_count
                columns[5], // release_date
                columns[6], // revenue
                columns[7], // runtime
                columns[15], // budget
                columns[16], // overview
                columns[17], // popularity
                columns[18], // tagline
                columns[19], // genres
                columns[20], // production_companies
                columns[23] // keywords
        };

        // Join the selected columns into a single output line
        String outputLine = String.join(",", requiredColumns);

        // Write the key-value pair to context (key is line offset, value is the
        // processed line)
        context.write(key, new Text(outputLine));
    }
}