# Use an official Node.js image as the base image
FROM node:latest

# Set the working directory in the container
WORKDIR /app

# Copy package.json and package-lock.json to the working directory
COPY package*.json ./

# Install dependencies
RUN npm install
# RUN npm install -g serve

# Copy the remaining application code to the working directory
COPY . .

# Build the React app for production
#RUN npm run build

# Expose port 3000
EXPOSE 3000

# Command to run the application
CMD ["npm", "start"]
#CMD ["serve", "-s", "build"]
